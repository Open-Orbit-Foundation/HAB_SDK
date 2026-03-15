from model_wind import LaunchSite, Balloon, Payload, MissionProfile, Model, ETOPO1Terrain
from gfs_wind import GFSWind
from hrrr_wind import HRRRWind
import numpy as np
import pandas as pd
import time
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import matplotlib.pyplot as plt
import contextily as cx

pd.set_option('display.max_rows', None)

# Worker-local caches so each process only builds terrain / wind once.
_WORKER_TERRAIN = None
_WORKER_WIND = None
_WORKER_SIG = None


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


def mp_progress(done, total, start, bar_len=60):
    frac = done / total if total else 1.0
    filled = int(bar_len * frac)
    bar = "#" * filled + "-" * (bar_len - filled)
    elapsed = time.perf_counter() - start
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = (total - done) / rate if rate > 0 else float("inf")
    sys.stdout.write(
        f"\rProcessing {total} profiles. |{bar}| {done:>5}/{total}  ({100*frac:>3.0f}%)  "
        f"{rate:>5.2f} profiles/s  ETA {eta:>6.1f}s   "
    )
    sys.stdout.flush()



def chunked_indexed(profiles, chunk_size: int):
    """Yield lists of (idx, profile) of length <= chunk_size."""
    batch = []
    for i, p in enumerate(profiles):
        batch.append((i, p))
        if len(batch) >= chunk_size:
            yield batch
            batch = []
    if batch:
        yield batch



def build_wind(wind_kind: str, run_time_utc: str):
    wind_kind = wind_kind.strip().lower()

    if wind_kind == "gfs":
        return GFSWind(
            run_utc=run_time_utc,
            save_dir="./gfs_downloads",
            product="pgrb2.0p25",
            preload_hours=3,
            max_hours_total=24,
            sample_time_bin_s=60.0,
            sample_alt_bin_m=100.0,
            sample_latlon_decimals=6,
        )

    if wind_kind == "hrrr":
        gfs_fallback = GFSWind(
            run_utc=run_time_utc,
            save_dir="./gfs_downloads",
            product="pgrb2.0p25",
            preload_hours=3,
            max_hours_total=24,
            sample_time_bin_s=60.0,
            sample_alt_bin_m=100.0,
            sample_latlon_decimals=6,
        )
        return HRRRWind(
            run_utc=run_time_utc,
            save_dir="./hrrr_downloads",
            product="prs",
            fallback_wind=gfs_fallback,
            preload_hours=3,
            max_hours_total=18,
            sample_time_bin_s=60.0,
            sample_alt_bin_m=100.0,
            sample_latlon_decimals=6,
            verbose=False,
        )

    raise ValueError(f"Unsupported wind_kind: {wind_kind}")



def get_worker_resources(wind_kind: str, run_time_utc: str):
    global _WORKER_TERRAIN, _WORKER_WIND, _WORKER_SIG

    sig = (wind_kind.strip().lower(), str(run_time_utc))
    if _WORKER_SIG != sig or _WORKER_TERRAIN is None or _WORKER_WIND is None:
        _WORKER_TERRAIN = ETOPO1Terrain()
        _WORKER_WIND = build_wind(wind_kind, run_time_utc)
        _WORKER_SIG = sig

    return _WORKER_TERRAIN, _WORKER_WIND



def predictor_batch(batch, dt, wind_kind, run_time_utc, logging=False):
    terrain, wind = get_worker_resources(wind_kind, run_time_utc)

    pairs = []
    for idx, profile in batch:
        out = []
        try:
            Model(
                time_step=dt,
                profiles=[profile],
                result=out,
                wind=wind,
                terrain=terrain,
                run_time_utc=run_time_utc,
            ).altitude_model(logging=logging)
            prof = out[0] if out else None
        except Exception:
            prof = None

        pairs.append((idx, prof))

    return pairs

def wrap_lon_180(lon_deg):
    return ((float(lon_deg) + 180.0) % 360.0) - 180.0

if __name__ == "__main__":
    # ---- Mission setup ----
    launch_site = LaunchSite(40.446387, -104.637853)

    fill_volumes = [150] #np.linspace(0, 300, 251)

    mission_profiles = []
    for v_fill in fill_volumes:
        b = Balloon(0.60, 6.02, 0.55, "Helium", float(v_fill))  # Kaymont 600g
        # b = Balloon(0.80, 7.00, 0.55, "Helium", float(v_fill))  # Kaymont 800g
        # b = Balloon(1.00, 7.86, 0.55, "Helium", float(v_fill))  # Kaymont 1000g
        # b = Balloon(1.20, 8.63, 0.55, "Helium", float(v_fill))  # Kaymont 1200g
        # b = Balloon(1.50, 9.44, 0.55, "Helium", float(v_fill))  # Kaymont 1500g
        # b = Balloon(2.00, 10.54, 0.55, "Helium", float(v_fill)) # Kaymont 2000g
        # b = Balloon(3.00, 13.00, 0.55, "Helium", float(v_fill)) # Kaymont 3000g
        #b = Balloon(4.00, 15.06, 0.55, "Helium", float(v_fill))   # Kaymont 4000g
        p = Payload(2, 4 * 0.3048, 0.5)
        mission_profiles.append(MissionProfile(launch_site, b, p))

    flight_profiles = [None] * len(mission_profiles)

    # ---- Wind selection ----
    # Choose ONE wind source for the whole batch run.
    WIND_KIND = "gfs"   # "gfs" or "hrrr"
    RUN_TIME_UTC = "2026-01-10 00:00"

    dt = 0.25

    # ============================================================
    # SINGLEPROCESSING
    # Comment out this block if you want multiprocessing instead.
    # ============================================================
    start = time.perf_counter()

    terrain = ETOPO1Terrain()
    wind = build_wind(WIND_KIND, RUN_TIME_UTC)

    Model(
        time_step=dt,
        profiles=mission_profiles,
        result=flight_profiles,
        wind=wind,
        terrain=terrain,
        run_time_utc=RUN_TIME_UTC,
    ).altitude_model(logging=True)

    end = time.perf_counter()
    print(f"Singleprocessing {len(mission_profiles)} profiles in {end - start:.2f} seconds.")

    # ============================================================
    # BATCHED MULTIPROCESSING
    # Comment out this block if you want singleprocessing instead.
    # ============================================================
    '''max_workers = 4
    CHUNK_SIZE = 10

    batches = list(chunked_indexed(mission_profiles, CHUNK_SIZE))
    total = len(mission_profiles)
    done = 0

    start = time.perf_counter()
    mp_progress(0, total, start)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        submit = partial(
            predictor_batch,
            dt=dt,
            wind_kind=WIND_KIND,
            run_time_utc=RUN_TIME_UTC,
            logging=False,
        )
        futures = [executor.submit(submit, batch) for batch in batches]

        for fut in as_completed(futures):
            pairs = fut.result()
            for idx, prof in pairs:
                flight_profiles[idx] = prof
            done += len(pairs)
            mp_progress(done, total, start)

    end = time.perf_counter()
    sys.stdout.write("\n")
    sys.stdout.flush()
    print(f"Multiprocessing {total} profiles in {end - start:.2f} seconds.")'''

    # ---- Post-processing ----
    df = profiles_to_dataframe([p for p in flight_profiles if p is not None])

    if not df.empty:
        df["vbar0_mps"] = df["burst_altitude"] / df["burst_time"]
        df["ok_burst"] = df["burst_altitude"].notna() & df["burst_time"].notna() & (df["burst_time"] > 0)
        df["sane"] = (
            df["ok_burst"]
            & np.isfinite(df["vbar0_mps"])
            & (df["vbar0_mps"] > 0)
            & (df["vbar0_mps"] < 20)
            & (df["burst_altitude"] > 1000)
            & (df["burst_altitude"] < 60000)
        )

        # Convenience landing columns for quick checks / future plotting.
        df["landing_latitude"] = [fp.latitudes[-1] for fp in flight_profiles if fp is not None]
        df["landing_longitude"] = [wrap_lon_180(fp.longitudes[-1]) for fp in flight_profiles if fp is not None]
        df["landing_altitude"] = [fp.altitudes[-1] for fp in flight_profiles if fp is not None]
        df["landing_ground_altitude"] = [fp.ground_altitudes[-1] for fp in flight_profiles if fp is not None]

        print(f"Total profiles: {len(df)}")
        print(f"Successful bursts: {int(df['ok_burst'].sum())}")
        print(f"Plotted (sane): {int(df['sane'].sum())}")
        print(df[[
            "payload.mass",
            "balloon.gas_volume",
            "burst_altitude",
            "burst_time",
            "flight_time",
            "landing_latitude",
            "landing_longitude",
        ]])

    # ---- Landing scatter map ----
    sane_df = df[df["sane"]].copy()

    if not sane_df.empty:
        fig, ax = plt.subplots(figsize=(10, 10))

        launch_lat = mission_profiles[0].launch_site.latitude
        launch_lon = ((mission_profiles[0].launch_site.longitude + 180.0) % 360.0) - 180.0

        land_lat = sane_df["landing_latitude"].to_numpy(dtype=float)
        land_lon = ((sane_df["landing_longitude"].to_numpy(dtype=float) + 180.0) % 360.0) - 180.0

        # Landing points: one point per run
        ax.scatter(
            land_lon,
            land_lat,
            s=30,
            alpha=0.75,
            label="Landing locations",
        )

        # Launch point: show once
        ax.scatter(
            [launch_lon],
            [launch_lat],
            s=120,
            marker="*",
            label="Launch location",
            zorder=5,
        )

        all_lon = np.concatenate([[launch_lon], land_lon])
        all_lat = np.concatenate([[launch_lat], land_lat])

        lon_span = np.ptp(all_lon)
        lat_span = np.ptp(all_lat)

        # Avoid zero-span issues when points are tightly clustered
        lon_pad = max(0.01, 0.08 * lon_span)
        lat_pad = max(0.01, 0.08 * lat_span)

        ax.set_xlim(all_lon.min() - lon_pad, all_lon.max() + lon_pad)
        ax.set_ylim(all_lat.min() - lat_pad, all_lat.max() + lat_pad)

        cx.add_basemap(
            ax,
            crs="EPSG:4326",
            source=cx.providers.OpenStreetMap.Mapnik,
            attribution_size=6,
        )

        ax.set_title("Launch and Landing Locations")
        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Latitude (deg)")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
        ax.set_axisbelow(True)
        ax.legend()

        plt.tight_layout()
        plt.show()
