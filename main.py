from model import LaunchSite, Balloon, Payload, MissionProfile#, Model
import run
import numpy as np
import pandas as pd
import time
from functools import partial
from atmosphere import standardAtmosphere
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import matplotlib.ticker as tic
import sys

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
    sys.stdout.write(f"\rProcessing {total} profiles... |{bar}| {done:>5}/{total}  ({100*frac:>3.0f}%)  "
                     f"{rate:>5.2f} profiles/s  ETA {eta:>6.1f}s   ")
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

if __name__ == "__main__":
    launch_site = LaunchSite(0.0)  # MSL reference for the chart

    # --- Sweep ranges (first take; adjust after you see the envelope) ---
    fill_volumes = np.linspace(0, 500, 101)            # ft^3
    suspended_masses = np.linspace(0, 10, 101)         # kg

    mission_profiles = []

    for m_payload in suspended_masses:
        for v_fill in fill_volumes:
            b = Balloon(0.60, 6.02, 0.55, "Helium", float(v_fill))  #Kaymont 600g
            #b = Balloon(0.80, 7.00, 0.55, "Helium", float(v_fill))  #Kaymont 800g
            #b = Balloon(1.00, 7.86, 0.55, "Helium", float(v_fill))  #Kaymont 1000g
            #b = Balloon(1.20, 8.63, 0.55, "Helium", float(v_fill))  #Kaymont 1200g
            #b = Balloon(1.50, 9.44, 0.55, "Helium", float(v_fill))  #Kaymont 1500g
            #b = Balloon(2.00, 10.54, 0.55, "Helium", float(v_fill)) #Kaymont 2000g
            #b = Balloon(3.00, 13.00, 0.55, "Helium", float(v_fill)) #Kaymont 3000g
            p = Payload(m_payload, 4 * 0.3048, 0.5)
            mission_profiles.append(MissionProfile(launch_site, b, p))

    #flight_profiles = []

    flight_profiles = [None] * len(mission_profiles)

    # ---- BATCHED MULTIPROCESSING ----
    dt = 0.15
    max_workers = 16

    # start with 20–50; tune later
    CHUNK_SIZE = 25

    batches = list(chunked_indexed(mission_profiles, CHUNK_SIZE))
    total = len(mission_profiles)
    done = 0

    '''mission_profiles = [MissionProfile(
        LaunchSite(1422), 
        Balloon(0.60, 6.02, 0.55, "Helium", 125), 
        Payload(1.5, 4 * 0.3048, 0.5)
    )]

    flight_profiles = []

    start = time.perf_counter()
    Model(0.1, mission_profiles, flight_profiles).altitude_model(True, 1)
    end= time.perf_counter()
    print(f"Singleprocessing {int(len(mission_profiles))} Profiles in {end - start:.2f} seconds.")'''

    start = time.perf_counter()
    mp_progress(0, total, start)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        submit = partial(run.predictor_batch, dt=dt, logging=False, interval=100)
        futures = [executor.submit(submit, batch) for batch in batches]
        for fut in as_completed(futures):
            pairs = fut.result()  # list[(idx, FlightProfile|None)]
            for idx, prof in pairs:
                flight_profiles[idx] = prof
            done += len(pairs)
            mp_progress(done, total, start)
    end= time.perf_counter()
    sys.stdout.write("\033[?25h\n")
    sys.stdout.flush()
    print(f"Multiprocessing {total} Profiles in {end - start:.2f} seconds.")

    # Build dataframe (handles nested attrs like balloon.gas_volume, payload.mass, etc.)
    df = profiles_to_dataframe([p for p in flight_profiles if p is not None])

    # Add reference-average ascent rate (MSL, USSA76): vbar0 = z_burst / t_burst
    # (Only defined when burst_time is finite & > 0)
    df["vbar0_mps"] = df["burst_altitude"] / df["burst_time"]

    # Mark "successful burst" runs
    df["ok_burst"] = df["burst_altitude"].notna() & df["burst_time"].notna() & (df["burst_time"] > 0)

    # A sanity filter for numerical blow-ups (keeps plots stable)
    # You can tighten/loosen these once you see real envelopes.
    df["sane"] = (
        df["ok_burst"]
        & np.isfinite(df["vbar0_mps"])
        & (df["vbar0_mps"] > 0)
        & (df["vbar0_mps"] < 20)          # ascent rate should not be anywhere near 20 m/s in normal HAB ops
        & (df["burst_altitude"] > 1000)   # ignore trivial "burst" near ground
        & (df["burst_altitude"] < 60000)  # ignore pathological overshoots
    )

    df_plot = df.loc[df["sane"]].copy()

    print(f"Total profiles: {len(df)}")
    print(f"Successful bursts: {df['ok_burst'].sum()}")
    print(f"Plotted (sane): {len(df_plot)}")

    # --- Compute initial net force at launch (neutral buoyancy boundary) ---
    atm = standardAtmosphere()
    launch_alt = float(df["launch_site.altitude"].iloc[0])  # should be 0 for your chart

    p0, T0, rho0, g0 = atm._Qualities(launch_alt)

    R_u = (1.380622 * 6.022169)  # same constant used in model.py
    # Volume at launch from moles under ambient conditions (match your model's formula)
    V0_m3 = df["balloon.gas_moles"] * R_u * T0 / p0 / 1000.0

    # Helium mass (kg) — same as in model
    m_He = df["balloon.gas_moles"] * 4.002602 / 1000.0

    m_total = df["payload.mass"] + df["balloon.mass"] + m_He

    Fnet0 = rho0 * g0 * V0_m3 - m_total * g0            # N
    a0 = Fnet0 / m_total                                 # m/s^2

    # Keep only finite points for contouring
    mask0 = np.isfinite(a0) & np.isfinite(df["payload.mass"]) & np.isfinite(df["balloon.gas_volume"])
    x0 = df.loc[mask0, "payload.mass"].to_numpy()
    y0 = df.loc[mask0, "balloon.gas_volume"].to_numpy()   # ft^3 in your current convention
    a0v = a0.loc[mask0].to_numpy()

    # --- Now build the plot (your existing contours) ---
    # Use df_plot for performance contours (burst altitude & vbar0), but use df (incl failures) for the infeasible shading
    x = df_plot["payload.mass"].to_numpy()
    y = df_plot["balloon.gas_volume"].to_numpy()
    z_alt_km = (df_plot["burst_altitude"] / 1000).to_numpy()
    z_vbar = df_plot["vbar0_mps"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 7))

    # 1) Shade infeasible region: a0 < 0 (initial downward acceleration)
    # Use a few levels so tricontourf can fill; the important boundary is 0.
    min_a = np.nanmin(a0v)
    levels_fill = [min_a, 0.0]
    ax.tricontourf(x0, y0, a0v, levels=levels_fill, alpha=0.18)

    # 2) Neutral buoyancy boundary: a0 = 0 (this is the curve you asked for)
    c0 = ax.tricontour(x0, y0, a0v, levels=[0.0], linewidths=2.2)
    ax.clabel(c0, fmt={0.0: "a₀ = 0"}, inline=True, fontsize=9)

    # 3) Your existing performance contours (burst altitude solid, vbar dashed)
    alt_levels = np.arange(np.floor(z_alt_km.min()), np.ceil(z_alt_km.max()) + 0.1, 1.0)  # 1 km
    v_levels = np.arange(np.floor(z_vbar.min()), np.ceil(z_vbar.max()) + 0.1, 0.5)        # 0.5 m/s

    cs1 = ax.tricontour(x, y, z_alt_km, levels=alt_levels, linewidths=1.5)
    ax.clabel(cs1, fmt="%.0f km", inline=True, fontsize=8)

    cs2 = ax.tricontour(x, y, z_vbar, levels=v_levels, linestyles="--", linewidths=1.0, alpha=0.9)
    ax.clabel(cs2, fmt="%.1f m/s", inline=True, fontsize=8)

    ax.set_title("Design Space (MSL, USSA76): Payload vs Fill\nContours: Burst Altitude (solid), $\\bar{v}_0$ (dashed), Neutral Buoyancy Boundary (bold)")
    ax.set_xlabel("Payload mass (kg)")
    ax.set_ylabel("Helium fill volume (ft³)")  # adjust if you change units later

    ax.xaxis.set_minor_locator(tic.AutoMinorLocator())
    ax.yaxis.set_minor_locator(tic.AutoMinorLocator())
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    '''fp = flight_profiles[-1]

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

    plt.show()'''