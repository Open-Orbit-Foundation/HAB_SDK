import numpy as np
import math
from dataclasses import dataclass

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
    pressures: list[float]
    #latitudes: list[float]
    #longitudes: list[float]

class Model:
    helium_mm = 4.002602
    stand_temp = 273.15
    stand_pres = 101.325
    gas_const = 1.380622 * 6.022169
    gas_vol_const = gas_const * stand_temp / stand_pres

    def __init__(self, time_step, profiles, result):
        self.time_step = time_step
        self.profiles = profiles
        self.result = result

    def altitude_model(self):
        for profile in self.profiles:
            altitudes = [profile.launch_site.altitude]
            pressure, temperature, density, gravity = profile.atmosphere._Qualities(altitudes[-1])
            pressures = [pressure]
            velocity = [0]
            acceleration = [0]
            times = [0]
            balloon_drag_coefficient = profile.balloon.drag_coefficient
            parachute_drag_coefficient = profile.payload.parachute_drag_coefficient
            parachute_area = math.pi / 4 * (profile.payload.parachute_diameter) ** 2
            burst_diameter = profile.balloon.burst_diameter
            burst_volume = 4 / 3 * math.pi * (burst_diameter / 2) ** 3
            gas_vol = profile.balloon.gas_volume * 0.3048 ** 3
            gas_moles = gas_vol * 1000 / self.gas_vol_const
            volume = gas_moles * self.gas_vol_const / 1000
            gas_mass = self.helium_mm * gas_moles / 1000
            ascent_mass = profile.payload.mass + profile.balloon.mass + gas_mass
            descent_mass = profile.payload.mass

            def ascent_accel(alt, v):
                pressure, temperature, density, gravity = profile.atmosphere._Qualities(alt)
                volume = gas_moles * self.gas_const * temperature / pressure / 1000
                area = math.pi * ((3 * volume) / (4 * math.pi)) ** (2 / 3)
                buoyant_force = density * gravity * volume
                weight_force = gravity * ascent_mass
                drag_force = 0.5 * density * v ** 2 * balloon_drag_coefficient * area * np.sign(v)
                net_force = buoyant_force - weight_force - drag_force
                return net_force / ascent_mass

            def descent_accel(alt, v):
                pressure, temperature, density, gravity = profile.atmosphere._Qualities(alt)
                weight_force = gravity * descent_mass
                drag_force = 0.5 * density * v ** 2 * parachute_drag_coefficient * parachute_area * np.sign(v)
                net_force = - weight_force - drag_force
                return net_force / descent_mass

            def rk4_update(alt, v, dt, accel_func):
                k1_v = accel_func(alt, v)
                k1_x = v
                k2_v = accel_func(alt + 0.5 * dt * k1_x, v + 0.5 * dt * k1_v)
                k2_x = v + 0.5 * dt * k1_v
                k3_v = accel_func(alt + 0.5 * dt * k2_x, v + 0.5 * dt * k2_v)
                k3_x = v + 0.5 * dt * k2_v
                k4_v = accel_func(alt + dt * k3_x, v + dt * k3_v)
                k4_x = v + dt * k3_v
                v_new = v + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
                alt_new = alt + (dt / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
                return alt_new, v_new, accel_func(alt_new, v_new)

            while volume < burst_volume:
                alt_new, v_new, a_new = rk4_update(altitudes[-1], velocity[-1], self.time_step, ascent_accel)
                if alt_new <= altitudes[-1]:
                    break
                acceleration.append(a_new)
                velocity.append(v_new)
                altitudes.append(alt_new)
                times.append(times[-1] + self.time_step)
                pressure, temperature, density, gravity = profile.atmosphere._Qualities(altitudes[-1])
                volume = gas_moles * self.gas_const * temperature / pressure / 1000  # m^3
                pressures.append(pressure)
            while altitudes[-1] >= altitudes[0]:  
                alt_new, v_new, a_new = rk4_update(altitudes[-1], velocity[-1], self.time_step, descent_accel)
                acceleration.append(a_new)
                velocity.append(v_new)
                altitudes.append(alt_new)
                times.append(times[-1] + self.time_step)
                pressure, temperature, density, gravity = profile.atmosphere._Qualities(altitudes[-1])
                pressures.append(pressure)
            altitudes.pop()
            velocity.pop()
            times.pop()
            pressures.pop()

            self.result.append(FlightProfile(profile.atmosphere, profile.launch_site, profile.balloon, profile.payload, times, altitudes, pressures))

        return self.result