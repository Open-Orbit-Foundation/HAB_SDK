from model_wind import LaunchSite, Balloon, Payload, MissionProfile, Model
from gfs_wind import GFSWind
from hrrr_wind import HRRRWind
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import contextily as cx
from pyproj import Transformer

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

    gfs = GFSWind(
        run_utc="2026-01-10 00:00",
        save_dir="./gfs_downloads",
        product="pgrb2.0p25",
        preload_hours=3,
        max_hours_total=24,
        sample_time_bin_s=60.0,
        sample_alt_bin_m=100.0,
        sample_latlon_decimals=6,
    )

    hrrr = HRRRWind(
        run_utc="2026-01-10 00:00",
        save_dir="./hrrr_downloads",
        product="prs",
        fallback_wind=gfs,
        preload_hours=3,
        max_hours_total=18,
        sample_time_bin_s=60.0,
        sample_alt_bin_m=100.0,
        sample_latlon_decimals=6,
        verbose=False
    )

    flight_profiles_gfs = []
    flight_profiles_hybrid = []

    Model(0.5, mission_profiles, flight_profiles_gfs, wind=gfs, run_time_utc="2026-01-10 00:00").altitude_model(True)

    Model(0.5, mission_profiles, flight_profiles_hybrid, wind=hrrr, run_time_utc="2026-01-10 00:00").altitude_model(True)

    fp_gfs = flight_profiles_gfs[-1]
    fp_hyb = flight_profiles_hybrid[-1]

    def extract(fp):
        t = np.asarray(fp.times, dtype=float)
        alt = np.asarray(fp.altitudes, dtype=float)
        vel = np.asarray(fp.velocities, dtype=float)  # (N,3)
        lat = np.asarray(fp.latitudes, dtype=float)
        lon = np.asarray(fp.longitudes, dtype=float)

        return {
            "t": t,
            "alt": alt,
            "vel": vel,
            "vz": vel[:, 2],
            "vxy": np.linalg.norm(vel[:, :2], axis=1),
            "v3": np.linalg.norm(vel, axis=1),
            "lat": lat,
            "lon": lon,
            "burst_alt": fp.burst_altitude,
            "burst_t": fp.burst_time,
        }

    g = extract(fp_gfs)
    h = extract(fp_hyb)

    '''def wrap180(deg):
        return (deg + 180.0) % 360.0 - 180.0

    heading_g = wrap180(np.degrees(np.arctan2(g["vel"][:,1], g["vel"][:,0])))
    heading_h = wrap180(np.degrees(np.arctan2(h["vel"][:,1], h["vel"][:,0])))

    plt.figure()
    plt.plot(g["t"]/60.0, heading_g, label="GFS heading")
    plt.plot(h["t"]/60.0, heading_h, label="HRRR→GFS heading", linestyle="--")
    plt.xlabel("Time (min)")
    plt.ylabel("Heading (deg, E=0, N=90)")
    plt.title("Horizontal Velocity Heading Comparison")
    plt.grid(True)
    plt.legend()'''

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # ----- Plot 1: Altitude vs time -----
    ax = axs[0,0]
    ax.plot(g["t"]/60.0, g["alt"], label="GFS")
    ax.plot(h["t"]/60.0, h["alt"], label="HRRR → GFS", linestyle="--")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Altitude (m)")
    ax.set_title("Altitude vs Time")
    ax.grid(True)
    ax.legend()

    # ----- Plot 2: Speeds vs time -----
    ax = axs[0,1]
    ax.plot(g["t"]/60.0, g["vz"], label="GFS Vz")
    ax.plot(h["t"]/60.0, h["vz"], label="HRRR→GFS Vz", linestyle="--")
    ax.plot(g["t"]/60.0, g["vxy"], label="GFS |Vxy|")
    ax.plot(h["t"]/60.0, h["vxy"], label="HRRR→GFS |Vxy|", linestyle="--")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Speed (m/s)")
    ax.set_title("Velocity Components")
    ax.grid(True)
    ax.legend()

    # ----- Plot 3: Ground track with OSM basemap, lat/lon axes -----
    ax = axs[1,0]

    g_lon = ((g["lon"] + 180.0) % 360.0) - 180.0
    h_lon = ((h["lon"] + 180.0) % 360.0) - 180.0

    ax.plot(g_lon, g["lat"], label="GFS", linewidth=2.0)
    ax.plot(h_lon, h["lat"], label="HRRR → GFS", linestyle="--", linewidth=2.0)

    all_lon = np.concatenate([g_lon, h_lon])
    all_lat = np.concatenate([g["lat"], h["lat"]])

    pad_frac = 0.08

    lon_span = np.ptp(all_lon)
    lat_span = np.ptp(all_lat)

    ax.set_xlim(all_lon.min() - pad_frac * lon_span,
                all_lon.max() + pad_frac * lon_span)
    ax.set_ylim(all_lat.min() - pad_frac * lat_span,
                all_lat.max() + pad_frac * lat_span)

    cx.add_basemap(
        ax,
        crs="EPSG:4326",
        source=cx.providers.OpenStreetMap.Mapnik,
        attribution_size=6,
    )

    ax.set_title("Ground Track (OSM)")
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend()

    u_g = np.asarray(fp_gfs.wind_u)
    v_g = np.asarray(fp_gfs.wind_v)
    u_h = np.asarray(fp_hyb.wind_u)
    v_h = np.asarray(fp_hyb.wind_v)

    spd_g = np.sqrt(u_g*u_g + v_g*v_g)
    spd_h = np.sqrt(u_h*u_h + v_h*v_h)

    # ----- Plot 4: Wind speed vs altitude -----
    ax = axs[1,1]
    ax.plot(spd_g, g["alt"], label="GFS wind speed")
    ax.plot(spd_h, h["alt"], label="HRRR wind speed", linestyle="--")
    ax.set_xlabel("Wind speed (m/s)")
    ax.set_ylabel("Altitude (m)")
    ax.set_title("Wind Speed vs Altitude")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()

    def haversine_km(lat1, lon1, lat2, lon2):
        R = 6371.0
        phi1 = np.radians(lat1); phi2 = np.radians(lat2)
        dphi = np.radians(lat2-lat1)
        dlmb = np.radians(lon2-lon1)
        a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlmb/2)**2
        return 2*R*np.arcsin(np.sqrt(a))

    g_end = (g["lat"][-1], ((g["lon"][-1]+180)%360)-180)
    h_end = (h["lat"][-1], ((h["lon"][-1]+180)%360)-180)

    print("Landing sep (km):", haversine_km(g_end[0], g_end[1], h_end[0], h_end[1]))