from utils import Geometry, Integrator, Utility
from dataclasses import dataclass
from functools import partial
import numpy as np
import math
import time

@dataclass(frozen=False)
class LaunchSite:
    altitude: float
    #latitude: float
    #longitude: float

@dataclass(frozen=False)
class Balloon:
    mass: float
    burst_diameter: float
    drag_coefficient: float
    gas: str
    gas_volume: float
    gas_moles: float = 0.0

    def __post_init__(self):
        self.gas_moles = self.gas_volume * 1000 / ((1.380622 * 6.022169) * 273.15 / 101.325)

@dataclass(frozen=False)
class Payload:
    mass: float
    parachute_diameter: float
    parachute_drag_coefficient: float

@dataclass(frozen=False)
class MissionProfile:
    atmosphere: object
    launch_site: LaunchSite
    balloon: Balloon
    payload: Payload

@dataclass(frozen=False)
class FlightProfile(MissionProfile):
    times: list[float]
    altitudes: list[float]
    velocities: list[float]
    accelerations: list[float]
    forces: list[float]
    pressures: list[float]
    temperatures: list[float]
    densities: list[float]
    gravities: list[float]
    #latitudes: list[float]
    #longitudes: list[float]

class Model:
    helium_mm = 4.002602

    def __init__(self, time_step, profiles, result):
        self.time_step = time_step
        self.profiles = profiles
        self.result = result

    def _acceleration(self, altitude, velocity, profile, ascent: bool):
                net_force = self._net_force(altitude, velocity, profile, ascent)
                if ascent:
                    mass = profile.payload.mass + profile.balloon.mass + (self.helium_mm * profile.balloon.gas_moles / 1000)
                else:
                    mass = profile.payload.mass
                return net_force / mass
    
    def _net_force(self, altitude, velocity, profile, ascent: bool):
                pressure, temperature, density, gravity = profile.atmosphere._Qualities(altitude)
                if ascent:
                    volume = profile.balloon.gas_moles * (1.380622 * 6.022169) * temperature / pressure / 1000
                    mass = profile.payload.mass + profile.balloon.mass + (self.helium_mm * profile.balloon.gas_moles / 1000)
                    buoyant_force = density * gravity * volume
                    weight_force = gravity * mass
                    drag_force = 0.5 * 0.5 * density * velocity ** 2 * profile.balloon.drag_coefficient * Geometry.sphere_cross_section(volume) * np.sign(velocity)
                else:
                    buoyant_force = 0
                    mass = profile.payload.mass
                    weight_force = gravity * mass
                    drag_force = 0.5 * 0.5 * density * velocity ** 2 * profile.payload.parachute_drag_coefficient * (profile.payload.parachute_diameter ** 2 * math.pi / 4) * np.sign(velocity)
                return buoyant_force - weight_force - drag_force
    
    def altitude_model(self, logging: bool):
        start = time.perf_counter()
        for i, profile in enumerate(self.profiles):
            padding = len(f" Modelling {len(self.profiles)} Profiles ...")
            Utility.progress_bar(i, len(self.profiles), prefix=f" Modelling {len(self.profiles)} Profiles ...".ljust(padding), suffix="Complete", bar_length=100) if logging else None
            print() if logging else None
            times = [0]
            altitudes = [profile.launch_site.altitude]
            velocities = [0]
            accelerations = [0]
            forces = [0]
            pressure, temperature, density, gravity = profile.atmosphere._Qualities(altitudes[-1])
            pressures = [pressure]
            temperatures = [temperature]
            densities = [density]
            gravities = [gravity]

            burst_volume = (4 / 3) * math.pi * (profile.balloon.burst_diameter / 2) ** 3
            volume = profile.balloon.gas_moles * (1.380622 * 6.022169) * temperature / pressure / 1000

            while volume < burst_volume:
                altitude, velocity, acceleration = Integrator.rk4_second_order(altitudes[-1], velocities[-1], partial(self._acceleration, profile = profile, ascent = True), self.time_step)
                accelerations.append(acceleration)
                velocities.append(velocity)
                altitudes.append(altitude)
                forces.append(self._net_force(altitudes[-1], velocities[-1], profile, True))
                times.append(times[-1] + self.time_step)
                pressure, temperature, density, gravity = profile.atmosphere._Qualities(altitudes[-1])
                pressures.append(pressure)
                temperatures.append(temperature)
                densities.append(density)
                gravities.append(gravity)
                volume = profile.balloon.gas_moles * (1.380622 * 6.022169) * temperature / pressure / 1000
                Utility.progress_bar(volume, burst_volume, prefix=f"Ascent ...".rjust(padding), suffix="Complete", bar_length=100) if logging else None
            print() if logging else None
            while altitudes[-1] >= altitudes[0]:
                altitude, velocity, acceleration = Integrator.rk4_second_order(altitudes[-1], velocities[-1], partial(self._acceleration, profile = profile, ascent = False), self.time_step)
                accelerations.append(acceleration)
                velocities.append(velocity)
                altitudes.append(altitude)
                forces.append(self._net_force(altitudes[-1], velocities[-1], profile, False))
                times.append(times[-1] + self.time_step)
                pressure, temperature, density, gravity = profile.atmosphere._Qualities(altitudes[-1])
                pressures.append(pressure)
                temperatures.append(temperature)
                densities.append(density)
                gravities.append(gravity)
                Utility.progress_bar(altitudes[0], altitudes[-1], prefix=f"Descent ...".rjust(padding), suffix="Complete", bar_length=100) if logging else None
            print(f"\x1B[2F\r\x1B[J\x1B[1F") if logging else None
            self.result.append(FlightProfile(profile.atmosphere, profile.launch_site, profile.balloon, profile.payload, 
                                             times, altitudes, velocities, accelerations, forces, 
                                             pressures, temperatures, densities, gravities))
        end = time.perf_counter()
        Utility.progress_bar(1, 1, prefix=f" Modelling {len(self.profiles)} Profiles ...".ljust(padding), suffix=f"Complete in {end - start:.2f} seconds\n", bar_length=100) if logging else print(f"Compute Time: {round(end - start, 2)}")