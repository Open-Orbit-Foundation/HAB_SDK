from utils import Geometry, Integrator, Utility
from atmosphere import standardAtmosphere
from dataclasses import dataclass
from functools import partial
import numpy as np
import math
import sys
import time
from datetime import timedelta, datetime, timezone

R_E = 6356766

@dataclass(frozen=False)
class LaunchSite:
    altitude: float
    latitude: float
    longitude: float

@dataclass(frozen=False)
class Balloon:
    mass: float
    burst_diameter: float
    drag_coefficient: float
    gas: str
    gas_volume: float
    gas_moles: float = 0.0

    def __post_init__(self):
        self.gas_moles = self.gas_volume * 3.048 ** 3 / 22.413636 #ft^3 to L to mol

@dataclass(frozen=False)
class Payload:
    mass: float
    parachute_diameter: float
    parachute_drag_coefficient: float
    parachute_area: float = 0.0

    def __post_init__(self):
        self.parachute_area = self.parachute_diameter ** 2 * math.pi / 4

@dataclass(frozen=False)
class MissionProfile:
    launch_site: LaunchSite
    balloon: Balloon
    payload: Payload

@dataclass(frozen=False)
class FlightProfile(MissionProfile):
    times: list[float]
    latitudes: list[float]
    longitudes: list[float]
    altitudes: list[float]
    velocities: list[float]
    accelerations: list[float]
    forces: list[float]
    pressures: list[float]
    temperatures: list[float]
    densities: list[float]
    gravities: list[float]
    wind_u: list[float]   # east (m/s)
    wind_v: list[float]   # north (m/s)
    burst_altitude: float
    max_altitude: float
    burst_time: float
    flight_time: float

class Model:
    helium_mm = 4.002602
    atmosphere = standardAtmosphere()

    def __init__(self, time_step, profiles, result, wind=None, run_time_utc=None):
        self.time_step = time_step
        self.profiles = profiles
        self.result = result
        self.wind = wind

        if isinstance(run_time_utc, str):
            self.run_time_utc = datetime.strptime(run_time_utc, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
        else:
            self.run_time_utc = run_time_utc

    # -------------------------
    # Core 3-DOF acceleration
    # -------------------------

    def _acceleration(self, r, v, profile, mass, current_time, lat, lon, x_prev, y_prev, Cd, area, buoyant: bool):
        x, y, z = r
        vx, vy, vz = v

        dx = x - x_prev
        dy = y - y_prev

        lat_new = lat + math.degrees(dy / R_E)
        lon_new = lon + math.degrees(dx / (R_E * math.cos(math.radians(lat_new))))

        pressure, temperature, density, gravity = self.atmosphere._Qualities(z)

        # Wind (east, north)
        if self.wind:
            u_wind, v_wind = self.wind.uv(current_time, z, lat_new, lon_new)
        else:
            u_wind = v_wind = 0.0

        # Relative velocity (balloon - air)
        v_rel = np.array([
            vx - u_wind,
            vy - v_wind,
            vz
        ])
        v_rel_mag = np.linalg.norm(v_rel)

        # Drag (vector)
        drag = (
            -0.5 * density * Cd * area
            * v_rel_mag * v_rel
        )

        # Buoyancy & gravity (vertical only)
        if buoyant:
            volume = profile.balloon.gas_moles * (1.380622 * 6.022169) * temperature / pressure / 1000
            buoyancy = np.array([0.0, 0.0, density * gravity * volume])
        else:
            buoyancy = np.array([0.0, 0.0, 0.0])
        weight = np.array([0.0, 0.0, -gravity * mass])

        F = buoyancy + weight + drag
        return F / mass

    # -------------------------
    # Main simulation
    # -------------------------

    def altitude_model(self, logging=True):
        start = time.perf_counter()

        for profile in self.profiles:
            # Initial geographic state
            lat = profile.launch_site.latitude
            lon = profile.launch_site.longitude
            z0 = profile.launch_site.altitude

            # Local tangent-plane state
            r = np.array([0.0, 0.0, z0])
            v = np.array([0.0, 0.0, 0.0])

            times = [0.0]
            latitudes = [lat]
            longitudes = [lon]
            altitudes = [z0]
            velocities = [v.copy()]
            accelerations = [np.zeros(3)]
            forces = [np.zeros(3)]
            pressures = []
            temperatures = []
            densities = []
            gravities = []
            wind_u = []
            wind_v = []

            ascent_mass = (
                profile.payload.mass
                + profile.balloon.mass
                + self.helium_mm * profile.balloon.gas_moles / 1000
            )
            descent_mass = profile.payload.mass

            burst_volume = (4 / 3) * math.pi * (profile.balloon.burst_diameter / 2) ** 3
            burst_altitude = None
            burst_time = None

            t = 0.0
            x_prev = y_prev = 0.0

            current_time = (self.run_time_utc if self.run_time_utc else None)
            u_wind, v_wind = self.wind.uv(current_time, r[2], lat, lon) if self.wind else (0.0, 0.0)
            wind_u.append(float(u_wind))
            wind_v.append(float(v_wind))

            while True:
                current_time = (
                    self.run_time_utc + timedelta(seconds=t)
                    if self.run_time_utc else None
                )

                pressure, temperature, density, gravity = self.atmosphere._Qualities(r[2])
                pressures.append(pressure)
                temperatures.append(temperature)
                densities.append(density)
                gravities.append(gravity)

                # -------------------------
                # Flight phase selection
                # -------------------------
                # Phase 1: ascent (buoyant balloon)
                # Phase 2: descent (post-burst, parachute)
                if burst_altitude is None:
                    # Ascent phase: balloon buoyancy + drag
                    volume = profile.balloon.gas_moles * (1.380622 * 6.022169) * temperature / pressure / 1000
                    buoyant = True
                    if volume >= burst_volume:
                        # Capture burst, start descent
                        burst_altitude = r[2]
                        burst_time = t
                        buoyant = False
                        mass = descent_mass
                        Cd = profile.payload.parachute_drag_coefficient
                        area = profile.payload.parachute_area
                    else:
                        mass = ascent_mass
                        Cd = profile.balloon.drag_coefficient
                        area = Geometry.sphere_cross_section(volume)
                else:
                    # Descent phase: gravity + parachute drag only
                    buoyant = False
                    mass = descent_mass
                    Cd = profile.payload.parachute_drag_coefficient
                    area = profile.payload.parachute_area

                accel_fn = partial(
                    self._acceleration,
                    profile=profile,
                    mass=mass,
                    current_time=current_time,
                    lat=lat,
                    lon=lon,
                    x_prev=x_prev,
                    y_prev=y_prev,
                    Cd=Cd,
                    area=area,
                    buoyant=buoyant
                )

                r, v, a = Integrator.rk4_second_order(r, v, accel_fn, self.time_step)

                # Update geographic coordinates
                dx = r[0] - x_prev
                dy = r[1] - y_prev

                lat_new = lat + math.degrees(dy / R_E)
                lon_new = lon + math.degrees(dx / (R_E * math.cos(math.radians(lat_new))))
                lat, lon = lat_new, lon_new

                x_prev, y_prev = r[0], r[1]

                t += self.time_step

                current_time_next = (self.run_time_utc + timedelta(seconds=t)) if self.run_time_utc else None
                u_wind, v_wind = self.wind.uv(current_time_next, r[2], lat, lon) if self.wind else (0.0, 0.0)
                wind_u.append(float(u_wind))
                wind_v.append(float(v_wind))

                times.append(t)
                latitudes.append(lat)
                longitudes.append(lon)
                altitudes.append(r[2])
                velocities.append(v.copy())
                accelerations.append(a)
                forces.append(a * mass)

                if r[2] <= z0 and burst_altitude is not None:
                    break

            self.result.append(
                FlightProfile(
                    profile.launch_site,
                    profile.balloon,
                    profile.payload,
                    times,
                    latitudes,
                    longitudes,
                    altitudes,
                    velocities,
                    accelerations,
                    forces,
                    pressures,
                    temperatures,
                    densities,
                    gravities,
                    wind_u,
                    wind_v,
                    float('nan') if burst_altitude is None else burst_altitude,
                    float(np.max(altitudes)),
                    float('nan') if burst_time is None else burst_time,
                    float(times[-1]),
                )
            )

        if logging:
            end = time.perf_counter()
            print(f"Processed {len(self.profiles)} profile(s) in {end - start:.2f} seconds")