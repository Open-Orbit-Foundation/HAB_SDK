from model_new import LaunchSite, Balloon, Payload, MissionProfile, Model
from atmosphere import standardAtmosphere
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as tic

launch_sites = [
    LaunchSite(1400)
]

balloons = [
    Balloon(0.6, 19.8 * 0.3048, 0.55, "Helium", 1.5),
    Balloon(0.35, 19.8 * 0.3048, 0.55, "Helium", 1.2),
    Balloon(0.6, 19.8 * 0.3048, 0.55, "Helium", 1.2),
    Balloon(0.35, 19.8 * 0.3048, 0.55, "Helium", 1)
]

payloads = [
    Payload(0.6, 2 * 0.3048, 1.2),
    Payload(0.3, 2 * 0.3048, 1.2)
]

mission_profiles = [
    MissionProfile(standardAtmosphere(), launch_sites[0], balloons[0], payloads[0]),
    MissionProfile(standardAtmosphere(), launch_sites[0], balloons[1], payloads[0]),
    MissionProfile(standardAtmosphere(), launch_sites[0], balloons[2], payloads[0]),
    MissionProfile(standardAtmosphere(), launch_sites[0], balloons[3], payloads[0])
]

flight_profiles = []

start = time.perf_counter()
Model(1, mission_profiles, flight_profiles).altitude_model()
end = time.perf_counter()
print(f"Compute Time: {round(end - start, 2)}")

fig, ax = plt.subplots(figsize = (12, 9))
for i, profile in enumerate(flight_profiles):
    ax.plot(np.array(profile.times) / 3600, np.array(profile.altitudes) / 1000, label = f"Profile {i + 1}")
    ax.set_xlabel("Time (hr)")
    ax.xaxis.set_major_locator(tic.MultipleLocator(1))
    ax.xaxis.set_minor_locator(tic.AutoMinorLocator(5))
    ax.set_ylabel("Altitude (km)")
    ax.yaxis.set_major_locator(tic.MultipleLocator(5))
    ax.yaxis.set_minor_locator(tic.AutoMinorLocator(6))
    burst_index = profile.altitudes.index(max(profile.altitudes))
    burst_time = profile.times[burst_index]
    burst_altitude = profile.altitudes[burst_index]
    ax.plot(burst_time / 3600, burst_altitude / 1000, 'x', color='black')
    ax.annotate(
        f"{burst_altitude / 1000:.1f} km\n{burst_time / 3600:.1f} hr",
        xy = (burst_time / 3600, burst_altitude / 1000),
        xytext = (-20, 5),
        textcoords = 'offset points',
        ha = 'center',
        fontsize = 8,
        bbox = dict(facecolor = 'white', alpha = 0.7, edgecolor = 'none', boxstyle = 'round, pad = 0.1')
    )
plt.title("RK4 Interial Model Altitude Profile(s)")
plt.legend()
plt.grid(True, which='both', linestyle='--')
plt.show()

fig, ax = plt.subplots()
for i, profile in enumerate(flight_profiles):
    ax.plot(np.array(profile.times) / 3600, np.array(profile.pressures) * 10, label = f"Profile {i + 1}")
    ax.set_xlabel("Time (hr)")
    ax.xaxis.set_major_locator(tic.MultipleLocator(1))
    ax.xaxis.set_minor_locator(tic.AutoMinorLocator(5))
    ax.set_ylabel("Pressure (hPa)")
    ax.set_yscale('log')
plt.title("RK4 Interial Model Pressure Profile(s)")
plt.legend()
plt.grid(True, which='both', linestyle='--')
plt.show()