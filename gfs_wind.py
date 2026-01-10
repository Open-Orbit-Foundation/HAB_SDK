from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Tuple, Dict, Optional

import numpy as np
from herbie import Herbie
import threading

# USSA geopotential reference Earth radius (m)
R_USSA = 6_356_766.0


# -------------------------
# Utilities
# -------------------------

def _to_datetime_utc(t) -> datetime:
    if isinstance(t, datetime):
        if t.tzinfo is None:
            return t.replace(tzinfo=timezone.utc)
        return t.astimezone(timezone.utc)

    if isinstance(t, np.datetime64):
        ts = (t - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s")
        return datetime.fromtimestamp(float(ts), tz=timezone.utc)

    if isinstance(t, str):
        s = t.strip().replace("T", " ")
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H"):
            try:
                return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                pass
        raise ValueError(f"Unrecognized time format: {t}")

    raise TypeError(f"Unsupported time type: {type(t)}")

def _floor_to_gfs_cycle(dt: datetime) -> datetime:
    """Floor datetime to the most recent GFS cycle time (00/06/12/18Z)."""
    dt = dt.astimezone(timezone.utc)
    cycle_hour = (dt.hour // 6) * 6
    return dt.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)

def _latest_safe_run_dt(lag_hours: float = 4.0) -> datetime:
    """Pick a run time that is very likely to exist on the public S3 buckets.

    We can't truly *know* availability without probing the network, so we use a small
    latency buffer and then snap to the previous 6-hourly cycle.
    """
    now = datetime.now(timezone.utc) - timedelta(hours=float(lag_hours))
    return _floor_to_gfs_cycle(now)

def _lon_to_0_360(lon):
    lon = float(lon)
    return lon % 360.0 if lon < 0 else lon

def _interp1d(x, y, xq):
    if xq <= x[0]:
        return float(y[0])
    if xq >= x[-1]:
        return float(y[-1])
    i = int(np.searchsorted(x, xq) - 1)
    t = (xq - x[i]) / (x[i + 1] - x[i])
    return float(y[i] + t * (y[i + 1] - y[i]))

def _bilinear_weights(x, x0, x1, y, y0, y1):
    tx = (x - x0) / (x1 - x0)
    ty = (y - y0) / (y1 - y0)
    return (
        (1 - tx) * (1 - ty),
        tx * (1 - ty),
        (1 - tx) * ty,
        tx * ty,
    )

# -------------------------
# Data container
# -------------------------

@dataclass
class _WindFile:
    fxx: int
    valid_time: datetime
    lat: np.ndarray
    lon: np.ndarray
    z_geom: np.ndarray
    u: np.ndarray
    v: np.ndarray

# -------------------------
# Main interface
# -------------------------

class GFSWind:
    """
    Lazy, cadence-aware GFS wind interpolator.

    Design goals:
      - Never assume hourly availability (supports 1p00, 0p50, 0p25 products)
      - Load only the forecast hours actually needed
      - Cache aggressively for use inside tight ODE integration loops
      - Remain robust to missing forecast files

    Time interpolation is always performed between the nearest
    available forecast hours bracketing the requested time.
    """

    def __init__(
        self,
        run_utc: str | datetime | None,
        save_dir: str | Path,
        product: str = "pgrb2.0p25",
        preload_hours: int = 0, # preload_hours: optional warm-start to avoid first-step GRIB decode latency
        max_hours_total: int = 240,
        auto_extend_max_hours: int = 384,
        run_utc_lag_hours: float = 4.0,
        clamp_run_utc_to_latest: bool = True,
        # ---- speed knobs (cache bins) ----
        sample_time_bin_s: float = 20.0,
        sample_alt_bin_m: float = 50.0,
        sample_latlon_decimals: int = 4,
        # ---- cache sizes ----
        sample_cache_max: int = 200000,
        grid_cache_max: int = 50000,
    ):
        # Resolve run time.
        # - If run_utc is None or 'latest', use a safe "latest available" run (with lag buffer).
        # - If run_utc is in the future, optionally clamp it back to the latest available run.
        if run_utc is None or (isinstance(run_utc, str) and run_utc.strip().lower() in ("latest", "auto")):
            self.run_dt = _latest_safe_run_dt(run_utc_lag_hours)
        else:
            self.run_dt = _to_datetime_utc(run_utc)
            if clamp_run_utc_to_latest:
                latest_ok = _latest_safe_run_dt(run_utc_lag_hours)
                if self.run_dt > latest_ok:
                    self.run_dt = latest_ok

        self.run_utc_str = self.run_dt.strftime("%Y-%m-%d %H:%M")
        self.product = product
        self.fxx_step_hours = self._gfs_fxx_step_hours(self.product)
        self.save_dir = Path(save_dir).resolve()
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self._missing_fxx: set[int] = set()

        self.max_hours_total = int(max_hours_total)
        self.auto_extend_max_hours = int(auto_extend_max_hours)
        if self.auto_extend_max_hours < self.max_hours_total:
            self.auto_extend_max_hours = self.max_hours_total

        self._lock = threading.Lock()
        self._files: Dict[int, _WindFile] = {}

        self._lat: Optional[np.ndarray] = None
        self._lon: Optional[np.ndarray] = None
        self._lat_asc: Optional[bool] = None

        self._col_cache: Dict[tuple, tuple] = {}
        self._col_cache_max = 20000  # adjust; keeps memory bounded

        # Cache the FINAL sampled answer (FlightPredictor-style)
        self.sample_time_bin_s = float(sample_time_bin_s)
        self.sample_alt_bin_m = float(sample_alt_bin_m)
        self.sample_latlon_decimals = int(sample_latlon_decimals)
        self._sample_cache: Dict[tuple, tuple] = {}
        self._sample_cache_max = int(sample_cache_max)

        # Cache grid brackets + bilinear weights for lat/lon bins
        self._grid_cache: Dict[tuple, tuple] = {}
        self._grid_cache_max = int(grid_cache_max)

         # ---- cadence-aware preload ----
        if preload_hours > 0:
            step = max(1, int(self.fxx_step_hours))
            target = int(preload_hours)

            # Preload only fxx values aligned to this product's cadence.
            # Example: step=3, preload_hours=7 => 0,3,6
            for fxx in range(0, target + 1, step):
                # Robust: if exact fxx doesn't exist, jump to the next available upward.
                f_ok = self._next_available_fxx(fxx, direction=+1)
                self._ensure_loaded(f_ok)

    # -------------------------
    # Public API
    # -------------------------

    def uv(self, time_utc, alt_m, lat_deg, lon_deg) -> Tuple[float, float]:
        t_dt = _to_datetime_utc(time_utc)
        alt = float(alt_m)
        latq = float(lat_deg)
        lonq = _lon_to_0_360(lon_deg)

        f0, f1, tf = self._time_bracket(t_dt)
        self._ensure_loaded(f0)
        self._ensure_loaded(f1)

        # Cache the FINAL sampled answer (binned inputs)
        skey = self._bin_key(t_dt, alt, latq, lonq, f0, f1)
        hit = self._sample_cache.get(skey)
        if hit is not None:
            return hit

        # Cached grid brackets + bilinear weights
        j0, j1, k0, k1, w00, w10, w01, w11 = self._grid_bracket_and_weights(latq, lonq)

        def sample(fxx):

            def corner(j, k):
                z, u, v = self._get_col(fxx, j, k)
                return _interp1d(z, u, alt), _interp1d(z, v, alt)

            u00, v00 = corner(j0, k0)
            u10, v10 = corner(j0, k1)
            u01, v01 = corner(j1, k0)
            u11, v11 = corner(j1, k1)

            return (
                w00 * u00 + w10 * u10 + w01 * u01 + w11 * u11,
                w00 * v00 + w10 * v10 + w01 * v01 + w11 * v11,
            )

        u0, v0 = sample(f0)
        if f0 == f1:
            out = (u0, v0)
        else:
            u1, v1 = sample(f1)
            out = (u0 + tf * (u1 - u0), v0 + tf * (v1 - v0))

        if len(self._sample_cache) >= self._sample_cache_max:
            self._sample_cache.clear()
        self._sample_cache[skey] = out
        return out

    # -------------------------
    # Internals
    # -------------------------

    @staticmethod
    def _gfs_fxx_step_hours(product: str) -> int:
        """
        Forecast-hour cadence by GFS product.

        Stable rule:
        - pgrb2.0p25 : 1-hourly forecast hours (FH001, FH002, ...)
        - pgrb2.1p00 : 3-hourly forecast hours (FH003, FH006, ...)
        - pgrb2.0p50 : typically 3-hourly
        """
        p = (product or "").lower()
        if "1p00" in p:
            return 3
        if "0p50" in p:
            return 3
        if "0p25" in p:
            return 1
        return 1

    def _next_available_fxx(self, start_fxx: int, direction: int) -> int:
        """
        direction = +1 -> search upward for the next existing fxx
        direction = -1 -> search downward for the previous existing fxx
        """
        f = int(start_fxx)

        # clamp to configured horizon
        hi = int(self.auto_extend_max_hours)
        if f < 0:
            f = 0
        if f > hi:
            f = hi

        while 0 <= f <= hi:
            if f not in self._missing_fxx:
                try:
                    self._ensure_loaded(f)   # will call _load_one if needed
                    return f                 # success
                except Exception:
                    # mark missing so we never try again
                    self._missing_fxx.add(f)
            f += direction

        # last resort: try f000
        self._ensure_loaded(0)
        return 0

    def _bin_key(self, t_dt: datetime, alt: float, latq: float, lonq: float, f0: int, f1: int):
        """Quantize inputs so repeated calls during integration hit cache."""
        dt_s = (t_dt - self.run_dt).total_seconds()
        tb = 0 if dt_s <= 0 else int(dt_s // self.sample_time_bin_s)

        ab = int(alt // self.sample_alt_bin_m) if self.sample_alt_bin_m > 0 else int(alt)

        p = self.sample_latlon_decimals
        latb = round(latq, p)
        lonb = round(lonq, p)

        return (f0, f1, tb, ab, latb, lonb)

    def _grid_bracket_and_weights(self, latq: float, lonq: float):
        """Return (j0,j1,k0,k1,w00,w10,w01,w11) with caching on binned lat/lon."""
        p = self.sample_latlon_decimals
        latb = round(latq, p)
        lonb = round(lonq, p)
        key = (latb, lonb)

        hit = self._grid_cache.get(key)
        if hit is not None:
            return hit

        lat = self._lat
        lon = self._lon

        j0, j1 = self._bracket(lat, latq, ascending=self._lat_asc)
        k0, k1 = self._bracket(lon, lonq, ascending=True)

        lat0, lat1 = float(lat[j0]), float(lat[j1])
        lon0, lon1 = float(lon[k0]), float(lon[k1])

        if j0 == j1:
            lat1 = lat0 + 1e-12
        if k0 == k1:
            lon1 = lon0 + 1e-12

        w00, w10, w01, w11 = _bilinear_weights(lonq, lon0, lon1, latq, lat0, lat1)

        out = (j0, j1, k0, k1, w00, w10, w01, w11)

        if len(self._grid_cache) >= self._grid_cache_max:
            self._grid_cache.clear()
        self._grid_cache[key] = out
        return out

    def _get_col(self, fxx: int, j: int, k: int):
        key = (fxx, j, k)
        hit = self._col_cache.get(key)
        if hit is not None:
            return hit

        wf = self._files[fxx]
        col = (wf.z_geom[:, j, k], wf.u[:, j, k], wf.v[:, j, k])

        if len(self._col_cache) >= self._col_cache_max:
            # simple eviction: clear; cheap + effective
            self._col_cache.clear()

        self._col_cache[key] = col
        return col

    def _time_bracket(self, t_dt):
        dt_hr = (t_dt - self.run_dt).total_seconds() / 3600.0

        # before run -> f000 only
        if dt_hr <= 0:
            f0 = 0
            self._ensure_loaded(f0)
            return f0, f0, 0.0

        step = int(self.fxx_step_hours) if getattr(self, "fxx_step_hours", None) else 1
        if step <= 0:
            step = 1

        # cadence-aware bracket:
        #   0p25 -> step=1 => behaves like before
        #   1p00 -> step=3 => brackets (0,3), (3,6), ... avoiding FH001/FH002 probes
        f_floor = int(np.floor(dt_hr / step) * step)
        f_ceil  = int(np.ceil(dt_hr / step) * step)

        # find nearest available at/below and at/above (still a safety net)
        f0 = self._next_available_fxx(f_floor, direction=-1)
        f1 = self._next_available_fxx(f_ceil,  direction=+1)

        self._ensure_loaded(f0)
        self._ensure_loaded(f1)

        # If both ends collapse (e.g., exact on-grid time or only one file available)
        if f0 == f1:
            return f0, f1, 0.0

        t0 = self._files[f0].valid_time
        t1 = self._files[f1].valid_time
        tf = (t_dt - t0).total_seconds() / (t1 - t0).total_seconds()
        tf = max(0.0, min(1.0, tf))
        return f0, f1, tf

    def _ensure_loaded(self, fxx: int):
        if fxx in self._files:
            return

        with self._lock:
            if fxx in self._files:
                return
            self._files[fxx] = self._load_one(fxx)  # let exceptions bubble up

            if self._lat is None:
                self._lat = self._files[fxx].lat
                self._lon = self._files[fxx].lon
                self._lat_asc = self._lat[0] < self._lat[-1]

    def _load_one(self, fxx: int) -> _WindFile:
        search = r":(?:UGRD|VGRD|HGT|GH):\d+ mb"

        H = Herbie(
            self.run_utc_str,
            model="gfs",
            product=self.product,
            fxx=fxx,
            save_dir=str(self.save_dir),
        )
        H.download(search=search)
        ds = H.xarray(search=search, decode_timedelta=False, remove_grib = False)

        def pick(name):
            for k in ds.data_vars:
                if name.lower() in k.lower():
                    return ds[k]
            raise KeyError(f"{name} not found in {list(ds.data_vars)}")

        u = pick("u").values.astype(np.float32)
        v = pick("v").values.astype(np.float32)
        gh = pick("gh").values.astype(np.float32)

        lat = ds["latitude"].values.astype(float)
        lon = ds["longitude"].values.astype(float)
        lon = np.where(lon < 0, lon % 360.0, lon)

        z_geom = (R_USSA * gh) / (R_USSA - gh)

        # Ensure vertical axis is increasing with index ONCE (avoid per-call argsort)
        # Check one representative column
        if z_geom[0, 0, 0] > z_geom[-1, 0, 0]:
            z_geom = z_geom[::-1, :, :]
            u = u[::-1, :, :]
            v = v[::-1, :, :]

        if "valid_time" in ds.coords:
            vt = _to_datetime_utc(ds["valid_time"].values)
        else:
            vt = self.run_dt + timedelta(hours=fxx)

        return _WindFile(
            fxx=fxx,
            valid_time=vt,
            lat=lat,
            lon=lon,
            z_geom=z_geom,
            u=u,
            v=v,
        )

    @staticmethod
    def _bracket(grid, xq, ascending=True):
        if ascending:
            if xq <= grid[0]:
                return 0, 0
            if xq >= grid[-1]:
                return len(grid) - 1, len(grid) - 1
            i = np.searchsorted(grid, xq)
            return i - 1, i
        else:
            if xq >= grid[0]:
                return 0, 0
            if xq <= grid[-1]:
                return len(grid) - 1, len(grid) - 1
            r = grid[::-1]
            i = np.searchsorted(r, xq)
            return len(grid) - i - 1, len(grid) - i
