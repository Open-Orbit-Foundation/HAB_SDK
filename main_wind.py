from model_wind import (
    LaunchSite,
    Balloon,
    Payload,
    MissionProfile,
    Model,
    ETOPO1Terrain,
    GroundNode,
    compute_node_observations_batch,
)
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
from datetime import datetime, timedelta, timezone

pd.set_option('display.max_rows', None)

#cloc . --exclude-dir=__pycache__,.vscode,gfs_downloads,hrrr_downloads,mission_design_spaces,terrain_cache,.git

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

def parse_utc_time(t):
    if isinstance(t, str):
        return datetime.strptime(t, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    return t

def format_utc_time(t):
    return t.strftime("%Y-%m-%d %H:%M")

def nominal_cycle_time_utc(wind_kind: str, launch_time_utc):
    t = parse_utc_time(launch_time_utc)
    wind_kind = wind_kind.strip().lower()

    if wind_kind == "gfs":
        cycle_hour = (t.hour // 6) * 6
        cycle = t.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)
        return cycle

    if wind_kind == "hrrr":
        cycle = t.replace(minute=0, second=0, microsecond=0)
        return cycle

    raise ValueError(f"Unsupported wind_kind: {wind_kind}")

def resolve_wind_cycle_time_utc(wind_kind: str, launch_time_utc, terrain=None):
    last_error = None

    for cycle_time_utc in cycle_candidates_utc(wind_kind, launch_time_utc):
        try:
            wind = build_wind(wind_kind, cycle_time_utc)
            return cycle_time_utc, wind
        except Exception as e:
            last_error = e

    raise RuntimeError(
        f"Could not build {wind_kind} wind for launch_time_utc={launch_time_utc}"
    ) from last_error

def cycle_candidates_utc(wind_kind: str, launch_time_utc, max_fallbacks=4):
    nominal = nominal_cycle_time_utc(wind_kind, launch_time_utc)
    wind_kind = wind_kind.strip().lower()

    step = timedelta(hours=6) if wind_kind == "gfs" else timedelta(hours=1)

    for k in range(max_fallbacks + 1):
        yield format_utc_time(nominal - k * step)

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

def get_worker_resources(wind_kind: str, launch_time_utc):
    global _WORKER_TERRAIN, _WORKER_WIND, _WORKER_SIG

    nominal_cycle = format_utc_time(nominal_cycle_time_utc(wind_kind, launch_time_utc))

    if _WORKER_TERRAIN is None:
        _WORKER_TERRAIN = ETOPO1Terrain()

    if (
        _WORKER_WIND is None
        or _WORKER_SIG is None
        or _WORKER_SIG[0] != wind_kind.strip().lower()
        or _WORKER_SIG[1] != nominal_cycle
    ):
        resolved_cycle, wind = resolve_wind_cycle_time_utc(wind_kind, launch_time_utc)
        _WORKER_WIND = wind
        _WORKER_SIG = (wind_kind.strip().lower(), nominal_cycle, resolved_cycle)

    return _WORKER_TERRAIN, _WORKER_WIND

def predictor_batch(batch, dt, wind_kind, logging=False):
    pairs = []
    for idx, profile in batch:
        terrain, wind = get_worker_resources(wind_kind, profile.launch_time_utc)

        out = []
        try:
            Model(
                time_step=dt,
                profiles=[profile],
                result=out,
                wind=wind,
                terrain=terrain,
                run_time_utc=profile.launch_time_utc,
            ).altitude_model(logging=logging)
            prof = out[0] if out else None
        except Exception:
            prof = None

        pairs.append((idx, prof))

    return pairs

def profile_batch_key(profile, wind_kind: str):
    nominal_cycle = format_utc_time(
        nominal_cycle_time_utc(wind_kind, profile.launch_time_utc)
    )
    return (wind_kind.strip().lower(), nominal_cycle)

def grouped_indexed_profiles(profiles, wind_kind: str):
    groups = {}
    for idx, profile in enumerate(profiles):
        key = profile_batch_key(profile, wind_kind)
        groups.setdefault(key, []).append((idx, profile))
    return groups

def chunk_group(group, chunk_size: int):
    batch = []
    for item in group:
        batch.append(item)
        if len(batch) >= chunk_size:
            yield batch
            batch = []
    if batch:
        yield batch

def wrap_lon_180(lon_deg):
    return ((float(lon_deg) + 180.0) % 360.0) - 180.0

def run_profiles(
    mission_profiles,
    dt,
    wind_kind,
    use_multiprocessing=False,
    max_workers=4,
    chunk_size=10,
):
    flight_profiles = [None] * len(mission_profiles)

    groups = grouped_indexed_profiles(mission_profiles, wind_kind)
    batches = []
    for group in groups.values():
        batches.extend(chunk_group(group, chunk_size))

    total = len(mission_profiles)
    start = time.perf_counter()

    if not use_multiprocessing:
        done = 0
        for batch in batches:
            pairs = predictor_batch(
                batch,
                dt=dt,
                wind_kind=wind_kind,
                logging=False,
            )
            for idx, prof in pairs:
                flight_profiles[idx] = prof
            done += len(pairs)
            mp_progress(done, total, start)

        sys.stdout.write("\n")
        sys.stdout.flush()
        end = time.perf_counter()
        print(f"Singleprocessing {total} profiles in {end - start:.2f} seconds.")
        return flight_profiles

    done = 0
    mp_progress(0, total, start)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        submit = partial(
            predictor_batch,
            dt=dt,
            wind_kind=wind_kind,
            logging=False,
        )
        futures = [executor.submit(submit, batch) for batch in batches]

        for fut in as_completed(futures):
            pairs = fut.result()
            for idx, prof in pairs:
                flight_profiles[idx] = prof
            done += len(pairs)
            mp_progress(done, total, start)

    sys.stdout.write("\n")
    sys.stdout.flush()
    end = time.perf_counter()
    print(f"Multiprocessing {total} profiles in {end - start:.2f} seconds.")
    return flight_profiles

def expand_bounds_to_aspect(xmin, xmax, ymin, ymax, width, height):
    xspan = max(xmax - xmin, 1e-9)
    yspan = max(ymax - ymin, 1e-9)

    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)

    target_aspect = width / height
    data_aspect = xspan / yspan

    if data_aspect < target_aspect:
        # too tall/narrow -> expand x
        new_xspan = yspan * target_aspect
        xmin = cx - 0.5 * new_xspan
        xmax = cx + 0.5 * new_xspan
    else:
        # too wide/short -> expand y
        new_yspan = xspan / target_aspect
        ymin = cy - 0.5 * new_yspan
        ymax = cy + 0.5 * new_yspan

    return xmin, xmax, ymin, ymax

def time_range_utc_minutes(start, end, step_minutes):
    t0 = parse_utc_time(start)
    t1 = parse_utc_time(end)

    times = []
    t = t0
    while t <= t1:
        times.append(format_utc_time(t))
        t += timedelta(minutes=step_minutes)

    return times

if __name__ == "__main__":
    # ---- Mission setup ----
    launch_site = LaunchSite(40.446387, -104.637853)

    fill_volumes = [150] #np.linspace(0, 300, 251)

    launch_times_utc = time_range_utc_minutes("2026-03-19 00:00", "2026-03-19 12:00", 30)

    mission_profiles = []
    for launch_time_utc in launch_times_utc:
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
            mission_profiles.append(
                MissionProfile(
                    launch_site=launch_site,
                    balloon=b,
                    payload=p,
                    launch_time_utc=launch_time_utc,
                )
            )

    ground_nodes = [
        GroundNode(
            latitude=40.446387,
            longitude=-104.637853,
            altitude=None,
            name="Launch Site GS",
        ),
    ]

    # ---- Wind selection ----
    # Choose ONE wind source for the whole batch run.
    WIND_KIND = "gfs"   # "gfs" or "hrrr"
    DRAW_GROUND_TRACES = False
    DRAW_BASEMAP = True

    dt = 0.25

    flight_profiles = run_profiles(
        mission_profiles=mission_profiles,
        dt=dt,
        wind_kind=WIND_KIND,
        use_multiprocessing=True,
        max_workers=4,
        chunk_size=10,
    )

    postprocess_terrain = ETOPO1Terrain()
    node_observation_profiles = compute_node_observations_batch(
        flight_profiles=flight_profiles,
        ground_nodes=ground_nodes,
        terrain=postprocess_terrain,
    )

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
        df["burst_latitude"] = [fp.burst_latitude for fp in flight_profiles if fp is not None]
        df["burst_longitude"] = [wrap_lon_180(fp.burst_longitude) for fp in flight_profiles if fp is not None]

        print(f"Total profiles: {len(df)}")
        print(f"Successful bursts: {int(df['ok_burst'].sum())}")
        print(f"Plotted (sane): {int(df['sane'].sum())}")
        print(df[[
            "payload.mass",
            "balloon.gas_volume",
            "launch_time_utc",
            "burst_time",
            "burst_altitude",
            "burst_latitude",
            "burst_longitude",
            "flight_time",
            "landing_latitude",
            "landing_longitude",
        ]])

    node_df = profiles_to_dataframe([q for q in node_observation_profiles if q is not None])

    if not node_df.empty:
        print(f"Node observation profiles: {len(node_df)}")
        print(node_df[[
            "node_name",
            "flight_profile_index",
            "launch_time_utc",
            "node_latitude",
            "node_longitude",
            "node_altitude",
        ]])

    obs0 = node_observation_profiles[0]
    print(obs0.node_name)
    print(obs0.launch_time_utc)
    print(obs0.times[:5])
    print(obs0.ranges_m[:5])
    print(obs0.azimuths_deg[:5])
    print(obs0.elevations_deg[:5])

    print("test")
    
    # ---- Launch / Burst / Landing map ----
    sane_df = df[df["sane"]].copy()

    if not sane_df.empty:
        fig_w, fig_h = 10, 10
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        launch_lat = mission_profiles[0].launch_site.latitude
        launch_lon = wrap_lon_180(mission_profiles[0].launch_site.longitude)

        land_lat = sane_df["landing_latitude"].to_numpy(dtype=float)
        land_lon = sane_df["landing_longitude"].to_numpy(dtype=float)

        burst_lat = sane_df["burst_latitude"].to_numpy(dtype=float)
        burst_lon = sane_df["burst_longitude"].to_numpy(dtype=float)

        sane_profiles = [
            fp for fp, ok in zip([p for p in flight_profiles if p is not None], df["sane"].tolist())
            if ok
        ]

        all_lon_parts = [[launch_lon], land_lon, burst_lon]
        all_lat_parts = [[launch_lat], land_lat, burst_lat]

        if DRAW_GROUND_TRACES:
            for fp in sane_profiles:
                trace_lon = np.array([wrap_lon_180(x) for x in fp.longitudes], dtype=float)
                trace_lat = np.array(fp.latitudes, dtype=float)
                all_lon_parts.append(trace_lon)
                all_lat_parts.append(trace_lat)

        all_lon = np.concatenate(all_lon_parts)
        all_lat = np.concatenate(all_lat_parts)

        xmin = all_lon.min()
        xmax = all_lon.max()
        ymin = all_lat.min()
        ymax = all_lat.max()

        lon_span = xmax - xmin
        lat_span = ymax - ymin

        lon_pad = max(0.01, 0.08 * lon_span)
        lat_pad = max(0.01, 0.08 * lat_span)

        xmin -= lon_pad
        xmax += lon_pad
        ymin -= lat_pad
        ymax += lat_pad

        xmin, xmax, ymin, ymax = expand_bounds_to_aspect(
            xmin, xmax, ymin, ymax, fig_w, fig_h
        )

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")

        if DRAW_BASEMAP:
            try:
                cx.add_basemap(
                    ax,
                    crs="EPSG:4326",
                    source=cx.providers.OpenStreetMap.Mapnik,
                    attribution_size=6,
                )
            except Exception as e:
                print(f"Warning: basemap download failed: {e}")

        if DRAW_GROUND_TRACES:
            for i, fp in enumerate(sane_profiles):
                trace_lon = np.array([wrap_lon_180(x) for x in fp.longitudes], dtype=float)
                trace_lat = np.array(fp.latitudes, dtype=float)
                ax.plot(
                    trace_lon,
                    trace_lat,
                    linewidth=2.0,
                    alpha=0.9,
                    zorder=8,
                    label="Ground trace" if i == 0 else None,
                )

        ax.scatter(
            land_lon,
            land_lat,
            s=30,
            alpha=0.75,
            label="Landing locations",
            zorder=9,
        )

        ax.scatter(
            burst_lon,
            burst_lat,
            s=45,
            marker="^",
            alpha=0.9,
            label="Burst locations",
            zorder=10,
        )

        ax.scatter(
            [launch_lon],
            [launch_lat],
            s=120,
            marker="*",
            label="Launch location",
            zorder=11,
        )

        # ---- Time progression path (connect sequential launches) ----
        order = sane_df.index.to_numpy()

        time_land_lon = land_lon
        time_land_lat = land_lat

        ax.plot(
            time_land_lon,
            time_land_lat,
            linewidth=2.5,
            alpha=0.9,
            color="black",
            zorder=7,
            label="Landing progression",
        )

        ax.set_title("Launch, Burst, and Landing Locations")
        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Latitude (deg)")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
        ax.set_axisbelow(True)
        ax.legend()

        plt.tight_layout()
        plt.show()