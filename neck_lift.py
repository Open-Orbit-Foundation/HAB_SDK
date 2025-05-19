from model import LaunchSite, Balloon, Payload, MissionProfile
from atmosphere import standardAtmosphere

launch_sites = [
    LaunchSite(1426)
]

balloons = [
    Balloon(0.60, 6.03504, 0.55, "Helium", 2.12376) #75 cuft
]

payloads = [
    Payload(0.952544, 2 * 0.3048, 1.2)
]

mission_profiles = [
    MissionProfile(launch_sites[0], balloons[0], payloads[0])#,
]

profile = mission_profiles[0]

atmosphere = standardAtmosphere()

altitude  = 1426

pressure, temperature, density, gravity = atmosphere._Qualities(altitude)
volume = profile.balloon.gas_moles * (1.380622 * 6.022169) * temperature / pressure / 1000
mass = profile.balloon.mass + (4.002602 * profile.balloon.gas_moles / 1000)
buoyant_force = density * gravity * volume
weight_force = gravity * mass

neck_lift = (buoyant_force - weight_force) / gravity
net_lift = (neck_lift - profile.payload.mass) * 1000

print(neck_lift)
print(net_lift)