from model_wind import LaunchSite, Balloon, Payload, MissionProfile, Model
from gfs_wind import GFSWind
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def extract_attributes(obj, prefix=""):
    attributes = {}
    for attr in dir(obj):
        if not callable(getattr(obj, attr)) and not attr.startswith("__"):
            value = getattr(obj, attr)
            if hasattr(value, "__dict__"):
                nested_attrs = extract_attributes(value, prefix=f"{prefix}{attr}.")
                attributes.update(nested_attrs)
            else:
                attributes[f"{prefix}{attr}"] = value
    return attributes

def profiles_to_dataframe(profiles):
    profile_data = []
    for i, profile in enumerate(profiles):
        profile_dict = {"Profile ID": i}
        attributes = extract_attributes(profile)
        profile_dict.update(attributes)
        profile_data.append(profile_dict)
    profile_df = pd.DataFrame(profile_data)
    profile_df.set_index("Profile ID", inplace=True)
    return profile_df

if __name__ == "__main__":

    mission_profiles = [MissionProfile(
        LaunchSite(1422, 40.446387, 255.362147), 
        Balloon(0.60, 6.02, 0.55, "Helium", 125), 
        Payload(1.5, 4 * 0.3048, 0.5)
    )]

    flight_profiles = []

    wind = GFSWind(
        run_utc="2026-01-10 00:00",
        save_dir="./gfs_downloads",
        product="pgrb2.0p25",
        preload_hours=3,
        max_hours_total=24,
        sample_time_bin_s=60.0,
        sample_alt_bin_m=100.0,
        sample_latlon_decimals=4,
    )

    Model(0.1, mission_profiles, flight_profiles, wind=wind, run_time_utc="2026-01-10 00:00").altitude_model(True)

    fp = flight_profiles[-1]

    t = np.asarray(fp.times, dtype=float)
    alt = np.asarray(fp.altitudes, dtype=float)

    vel = np.asarray(fp.velocities, dtype=float)      # shape (N,3)

    lat = np.asarray(fp.latitudes, dtype=float)
    lon = np.asarray(fp.longitudes, dtype=float)

    # ----- derived quantities -----
    vz = vel[:, 2]
    vxy = np.linalg.norm(vel[:, :2], axis=1)
    v3 = np.linalg.norm(vel, axis=1)

    # burst markers (if available)
    burst_alt = fp.burst_altitude
    burst_t = fp.burst_time
    has_burst = np.isfinite(burst_alt) and np.isfinite(burst_t)

    # ----- Plot 1: Altitude vs time -----
    plt.figure()
    plt.plot(t/60.0, alt)
    if has_burst:
        plt.axvline(burst_t/60.0, linestyle="--")
        plt.title(f"Altitude vs Time (burst @ {burst_t/60.0:.1f} min, {burst_alt:.0f} m)")
    else:
        plt.title("Altitude vs Time (no burst recorded)")
    plt.xlabel("Time (min)")
    plt.ylabel("Altitude (m)")
    plt.grid(True)

    # ----- Plot 2: Speeds vs time -----
    plt.figure()
    plt.plot(t/60.0, vz, label="Vertical Speed V_z (m/s)")
    plt.plot(t/60.0, vxy, label="Horizontal Speed |V_xy| (m/s)")
    plt.plot(t/60.0, v3, label="Total Speed |V| (m/s)")
    if has_burst:
        plt.axvline(burst_t/60.0, linestyle="--")
    plt.xlabel("Time (min)")
    plt.ylabel("Speed (m/s)")
    plt.title("Velocity Components")
    plt.grid(True)
    plt.legend()

    # ----- Plot 3: Ground track (lon/lat) -----
    plt.figure()
    lon_plot = ((lon + 180.0) % 360.0) - 180.0
    plt.plot(lon_plot, lat)
    plt.xlabel("Longitude (deg)")
    plt.ylabel("Latitude (deg)")
    plt.title("Ground Track")
    plt.grid(True)

    plt.show()