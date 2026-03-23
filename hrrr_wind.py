from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple
import threading

import numpy as np
from herbie import Herbie

import xarray as xr
xr.set_options(use_new_combine_kwarg_defaults=True)

# --- Constants ---
R_USSA = 6356766.0  # US Standard Atmosphere 1976 Earth radius (m)

def _floor_to_hour(dt: datetime) -> datetime:
    dt = dt.astimezone(timezone.utc)
    return dt.replace(minute=0, second=0, microsecond=0)

def _latest_safe_hrrr_run_dt(lag_hours: float = 2.0) -> datetime:
    # HRRR is hourly; give it a small buffer so we don't request a run not yet on the bucket.
    now = datetime.now(timezone.utc) - timedelta(hours=float(lag_hours))
    return _floor_to_hour(now)

def _to_datetime_utc(t) -> datetime:
    """Accept datetime / numpy datetime64 / pandas Timestamp / ISO string, return tz-aware UTC datetime."""
    if isinstance(t, datetime):
        return t if t.tzinfo is not None else t.replace(tzinfo=timezone.utc)

    # numpy datetime64 (treated as UTC by convention)
    try:
        import numpy as _np
        if isinstance(t, _np.datetime64):
            ts = (t - _np.datetime64("1970-01-01T00:00:00")) / _np.timedelta64(1, "s")
            return datetime.fromtimestamp(float(ts), tz=timezone.utc)
    except Exception:
        pass

    # pandas Timestamp
    try:
        import pandas as _pd
        if isinstance(t, _pd.Timestamp):
            dt = t.to_pydatetime()
            return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        pass

    if isinstance(t, str):
        dt = datetime.fromisoformat(t.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

    raise TypeError(f"Unsupported time type: {type(t)}")


def _interp1d(x: np.ndarray, y: np.ndarray, xq: float) -> float:
    """Linear interpolate y(x) at xq with clamping."""
    if xq <= x[0]:
        return float(y[0])
    if xq >= x[-1]:
        return float(y[-1])
    i = int(np.searchsorted(x, xq))
    i0 = i - 1
    i1 = i
    x0 = float(x[i0]); x1 = float(x[i1])
    if x1 == x0:
        return float(y[i0])
    w = (xq - x0) / (x1 - x0)
    return float((1.0 - w) * y[i0] + w * y[i1])


@dataclass(frozen=True)
class _WindFile:
    fxx: int
    valid_time: datetime
    x: np.ndarray               # (nx,)
    y: np.ndarray               # (ny,)
    x_asc: bool
    y_asc: bool
    z_geom: np.ndarray          # (nz, ny, nx) geometric height (m)
    u: np.ndarray               # (nz, ny, nx) grid-relative or earth-relative depending on uv_grid_relative
    v: np.ndarray               # (nz, ny, nx)
    z_max_m: float
    uv_grid_relative: bool


class HRRRWind:
    def __init__(
        self,
        run_utc: str | datetime,
        save_dir: str = "./hrrr_downloads",
        product: str = "prs",
        fallback_wind=None,
        preload_hours: int = 3,
        max_hours_total: int = 18,
        sample_time_bin_s: float = 60.0,
        sample_alt_bin_m: float = 100.0,
        sample_latlon_decimals: int = 4,
        verbose: bool = False,
        run_utc_lag_hours: float = 2.0,
        clamp_run_utc_to_latest: bool = True,
    ):
        """HRRR wind interpolator.

        Behavior intentionally matches `GFSWind` semantics:

        - `run_utc` is treated as the *requested valid time* you care about (typically your sim start time).
        - We choose the HRRR analysis cycle (`self.run_dt`) as the hour-aligned cycle at/before `run_utc`.
        - If that cycle would be in the future / not yet available, we clamp `self.run_dt` back to the latest
          safe cycle (using a lag buffer).
        - Preload starts at the forecast hour that corresponds to `run_utc` relative to `self.run_dt`
          (e.g., requested 06z with latest available cycle 02z => preload begins at F04, not F00).

        Notes:
          - HRRR forecast cadence is hourly (step=1).
          - When clamped, we keep the original requested time to compute `f_start`, so behavior stays consistent
            even if you asked for a future valid time.
        """

        if run_utc is None:
            raise ValueError("HRRRWind requires run_utc (a requested valid time).")

        # Requested valid time (what you asked for / sim start time)
        requested_dt = _to_datetime_utc(run_utc)

        # HRRR cycles hourly: the desired cycle is the hour-aligned requested time.
        desired_cycle = _floor_to_hour(requested_dt)

        # Clamp future/unavailable cycles back to a safe latest cycle if requested.
        self.run_dt = desired_cycle
        if clamp_run_utc_to_latest:
            latest_ok = _latest_safe_hrrr_run_dt(run_utc_lag_hours)
            if self.run_dt > latest_ok:
                self.run_dt = latest_ok

        self.run_utc_str = self.run_dt.strftime("%Y-%m-%d %H:%M")

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.product = product
        self.fallback_wind = fallback_wind

        self.preload_hours = int(preload_hours)
        self.max_hours_total = int(max_hours_total)

        self.sample_time_bin_s = float(sample_time_bin_s)
        self.sample_alt_bin_m = float(sample_alt_bin_m)
        self.sample_latlon_decimals = int(sample_latlon_decimals)
        self.verbose = bool(verbose)

        self._lock = threading.RLock()
        self._files: Dict[int, _WindFile] = {}
        self._sample_cache: Dict[Tuple, Tuple[float, float]] = {}

        # Projection objects (initialized on first file load)
        self._crs_grid = None
        self._proj = None              # pyproj.Proj
        self._ll_to_xy = None          # pyproj.Transformer
        self._xy_to_ll = None
        self._gamma_cache: Dict[Tuple[float, float], Tuple[float, float]] = {}  # (lat,lon)->(cosg,sing)
        self._printed_fileinfo = False
        self._printed_convergence = False

        # ---- cadence-aware preload (hourly HRRR) ----
        if self.preload_hours > 0:
            dt_hr = (requested_dt - self.run_dt).total_seconds() / 3600.0
            f_start = int(np.floor(dt_hr))  # hourly cadence
            f_start = max(0, min(f_start, self.max_hours_total))

            f0 = f_start
            fN = min(self.max_hours_total, f_start + self.preload_hours)

            for fxx in range(f0, fN + 1):
                self._ensure_loaded(fxx)

    # ---------- Public API ----------
    def uv(self, time_utc, alt_m, lat_deg, lon_deg) -> Tuple[float, float]:
        t_dt = _to_datetime_utc(time_utc)
        alt = float(alt_m)
        latq = float(lat_deg)
        lonq = self._wrap180(float(lon_deg))

        f0, f1, tf = self._time_bracket(t_dt)
        self._ensure_loaded(f0)
        self._ensure_loaded(f1)

        # Altitude ceiling check (avoid extrapolation)
        zmax = min(self._files[f0].z_max_m, self._files[f1].z_max_m)
        if alt > zmax:
            if self.fallback_wind is not None:
                return self.fallback_wind.uv(t_dt, alt, latq, lonq)
            alt = zmax

        # Cache key
        skey = self._bin_key(t_dt, alt, latq, lonq, f0, f1)
        hit = self._sample_cache.get(skey)
        if hit is not None:
            return hit

        # Project query point to grid x/y (meters)
        xq, yq = self._project_lonlat(lonq, latq)

        def sample_one(fxx: int) -> Tuple[float, float]:
            wf = self._files[fxx]
            j0, j1, k0, k1, w00, w10, w01, w11 = self._xy_bracket_and_weights(wf, xq, yq)

            def corner(j, k):
                zcol = wf.z_geom[:, j, k]
                ucol = wf.u[:, j, k]
                vcol = wf.v[:, j, k]
                ug = _interp1d(zcol, ucol, alt)
                vg = _interp1d(zcol, vcol, alt)
                if wf.uv_grid_relative:
                    ue, ve = self._grid_to_earth(ug, vg, lonq, latq)
                    return ue, ve
                return ug, vg

            u00, v00 = corner(j0, k0)
            u10, v10 = corner(j0, k1)
            u01, v01 = corner(j1, k0)
            u11, v11 = corner(j1, k1)

            uxy = w00 * u00 + w10 * u10 + w01 * u01 + w11 * u11
            vxy = w00 * v00 + w10 * v10 + w01 * v01 + w11 * v11
            return float(uxy), float(vxy)

        u0, v0 = sample_one(f0)
        if f1 != f0:
            u1, v1 = sample_one(f1)
            out = (float((1.0 - tf) * u0 + tf * u1), float((1.0 - tf) * v0 + tf * v1))
        else:
            out = (u0, v0)

        self._sample_cache[skey] = out
        return out

    # ---------- Internals ----------
    def _gamma_numeric_deg(self, lon_deg: float, lat_deg: float, dx_m: float = 1000.0) -> float:
        """
        Numerically estimate gamma (deg) by stepping +dx_m in grid-x and measuring
        the resulting direction in geographic coordinates.

        Returns gamma in the SAME convention used by _grid_to_earth's rotation matrix:
            u_e = u_g*cos(gamma) - v_g*sin(gamma)
            v_e = u_g*sin(gamma) + v_g*cos(gamma)
        """
        if self._ll_to_xy is None or self._xy_to_ll is None:
            self._ensure_loaded(0)

        x0, y0 = self._ll_to_xy.transform(float(lon_deg), float(lat_deg))
        lon1, lat1 = self._xy_to_ll.transform(float(x0 + dx_m), float(y0))

        # Bearing from (lat,lon) to (lat1,lon1), radians from north
        phi0 = np.deg2rad(float(lat_deg))
        phi1 = np.deg2rad(float(lat1))
        dlon = np.deg2rad(((float(lon1) - float(lon_deg) + 540.0) % 360.0) - 180.0)

        y = np.sin(dlon) * np.cos(phi1)
        x = np.cos(phi0) * np.sin(phi1) - np.sin(phi0) * np.cos(phi1) * np.cos(dlon)
        bearing_from_north = np.arctan2(y, x)

        # Convert to angle from east (CCW positive): angle_e = pi/2 - bearing
        angle_from_east = (np.pi / 2.0) - bearing_from_north

        # We want gamma = angle from grid-x TO east (to match your rotation matrix),
        # but angle_from_east is east->gridx, so negate:
        gamma = -angle_from_east

        return float(np.rad2deg(gamma))

    def _wrap180(self, lon):
        return (lon + 180.0) % 360.0 - 180.0

    def _bin_key(self, t_dt: datetime, alt: float, latq: float, lonq: float, f0: int, f1: int):
        dt_s = (t_dt - self.run_dt).total_seconds()
        tb = 0 if dt_s <= 0 else int(dt_s // self.sample_time_bin_s)
        ab = int(alt // self.sample_alt_bin_m) if self.sample_alt_bin_m > 0 else int(alt)

        p = self.sample_latlon_decimals
        latb = round(latq, p)
        lonb = round(lonq, p)
        return (f0, f1, tb, ab, latb, lonb)

    def _time_bracket(self, t_dt: datetime) -> Tuple[int, int, float]:
        dt_hr = (t_dt - self.run_dt).total_seconds() / 3600.0
        if dt_hr <= 0:
            return 0, 0, 0.0

        f0 = int(np.floor(dt_hr))
        f1 = int(np.ceil(dt_hr))
        f0 = self._clamp_fxx(f0)
        f1 = self._clamp_fxx(f1)

        if f0 == f1:
            return f0, f1, 0.0

        t0 = self.run_dt + timedelta(hours=f0)
        t1 = self.run_dt + timedelta(hours=f1)
        tf = (t_dt - t0).total_seconds() / (t1 - t0).total_seconds()
        tf = float(np.clip(tf, 0.0, 1.0))
        return f0, f1, tf

    def _clamp_fxx(self, fxx: int) -> int:
        if fxx < 0:
            return 0
        if fxx > self.max_hours_total:
            return self.max_hours_total
        return fxx

    def _ensure_loaded(self, fxx: int) -> None:
        with self._lock:
            if fxx in self._files:
                return
        wf = self._load_one(fxx)
        with self._lock:
            self._files[fxx] = wf

    def _ensure_projection(self, ds) -> None:
        """Initialize CRS/projection/transformer from gribfile_projection."""
        if self._ll_to_xy is not None and self._proj is not None:
            return

        try:
            import pyproj
        except Exception as e:
            raise RuntimeError(
                "Projected HRRR interpolation requires pyproj. Install with: conda install -c conda-forge pyproj"
            ) from e

        proj_var = None
        if "gribfile_projection" in ds.coords:
            proj_var = ds.coords["gribfile_projection"]
        elif "gribfile_projection" in ds.variables:
            proj_var = ds["gribfile_projection"]

        attrs = getattr(proj_var, "attrs", {}) if proj_var is not None else {}

        if "crs_wkt" in attrs:
            crs_grid = pyproj.CRS.from_wkt(attrs["crs_wkt"])
        else:
            # fall back to GRIB attrs on u/v if WKT missing
            crs_grid = pyproj.CRS.from_proj4(self._infer_proj4_from_attrs(attrs, ds.attrs))

        self._crs_grid = crs_grid
        self._proj = pyproj.Proj(crs_grid)
        self._ll_to_xy = pyproj.Transformer.from_crs(pyproj.CRS.from_epsg(4326), crs_grid, always_xy=True)
        self._xy_to_ll = pyproj.Transformer.from_crs(crs_grid, pyproj.CRS.from_epsg(4326), always_xy=True)


    def _load_one(self, fxx: int) -> _WindFile:
        """
        Download and load a single HRRR forecast hour using Herbie + cfgrib/xarray,
        extracting u/v and geopotential height on isobaric levels.
        """
        search = r":(?:UGRD|VGRD|HGT|GH):\d+ mb"

        H = Herbie(
            self.run_utc_str,
            model="hrrr",
            product=self.product,
            fxx=fxx,
            save_dir=str(self.save_dir),
            verbose=self.verbose,
        )
        H.download(search=search)
        ds = H.xarray(search=search, decode_timedelta=False, remove_grib=False)

        self._ensure_projection(ds)

        def pick_da(kind: str):
            if kind == "u":
                candidates = ["u", "UGRD", "u_component_of_wind"]
            elif kind == "v":
                candidates = ["v", "VGRD", "v_component_of_wind"]
            else:
                candidates = ["hgt", "HGT", "gh", "GH", "z", "geopotential_height"]

            for c in candidates:
                if c in ds:
                    return ds[c]

            # Fallback by GRIB shortName
            for name in ds.data_vars:
                a = ds[name].attrs
                sn = str(a.get("GRIB_shortName", "")).lower()
                if kind == "u" and sn in ("ugrd", "u"):
                    return ds[name]
                if kind == "v" and sn in ("vgrd", "v"):
                    return ds[name]
                if kind == "h" and sn in ("hgt", "gh", "z"):
                    return ds[name]
            raise KeyError(f"Could not find HRRR variable for {kind}. Vars: {list(ds.data_vars)}")

        u_da = pick_da("u")
        v_da = pick_da("v")
        h_da = pick_da("h")

        uv_grid_relative = bool(int(u_da.attrs.get("GRIB_uvRelativeToGrid", 0)) == 1)

        # load arrays
        u = np.asarray(u_da.values, dtype=np.float32)
        v = np.asarray(v_da.values, dtype=np.float32)
        h = np.asarray(h_da.values, dtype=np.float32)

        # Build x/y vectors (meters)
        x_vec, y_vec = self._pick_xy(ds, u_da.attrs)

        x_vec = np.asarray(x_vec, dtype=np.float64)
        y_vec = np.asarray(y_vec, dtype=np.float64)

        x_asc = bool(x_vec[0] < x_vec[-1])
        y_asc = bool(y_vec[0] < y_vec[-1])

        # Convert geopotential height to geometric height
        z_geom = (R_USSA * h) / (R_USSA - h)

        # Ensure vertical axis increasing
        if z_geom[0, 0, 0] > z_geom[-1, 0, 0]:
            z_geom = z_geom[::-1, :, :]
            u = u[::-1, :, :]
            v = v[::-1, :, :]

        z_max = float(np.nanmax(z_geom))

        # valid time
        if "valid_time" in ds.coords:
            vt = _to_datetime_utc(ds["valid_time"].values)
        else:
            vt = self.run_dt + timedelta(hours=fxx)

        # Optional one-time debug print to confirm rotation path
        if self.verbose and (not self._printed_fileinfo):
            print(f"[HRRR] f{fxx:02d} uv_grid_relative={uv_grid_relative}", flush=True)
            self._printed_fileinfo = True

        return _WindFile(
            fxx=fxx,
            valid_time=vt,
            x=x_vec,
            y=y_vec,
            x_asc=x_asc,
            y_asc=y_asc,
            z_geom=z_geom,
            u=u,
            v=v,
            z_max_m=z_max,
            uv_grid_relative=uv_grid_relative,
        )

    def _pick_xy(self, ds, var_attrs: dict):
        """Return 1D x and y coordinate vectors (meters)."""
        # Prefer explicit coords
        if "x" in ds.coords and "y" in ds.coords:
            return ds["x"].values, ds["y"].values

        # Search for CF standard_name coords
        x_name = None
        y_name = None
        for cname, c in ds.coords.items():
            sn = str(c.attrs.get("standard_name", ""))
            if sn == "projection_x_coordinate":
                x_name = cname
            elif sn == "projection_y_coordinate":
                y_name = cname
        if x_name and y_name:
            return ds[x_name].values, ds[y_name].values

        # Fallback: synthesize from 2D lat/lon and Dx/Dy
        if "latitude" not in ds.coords or "longitude" not in ds.coords:
            raise KeyError(f"Could not find HRRR x/y projection coordinates or latitude/longitude in coords: {list(ds.coords)}")

        lat2 = np.asarray(ds["latitude"].values, dtype=np.float64)
        lon2 = np.asarray(ds["longitude"].values, dtype=np.float64)
        if lat2.ndim != 2 or lon2.ndim != 2:
            raise KeyError(f"Expected 2D latitude/longitude; got lat={lat2.shape}, lon={lon2.shape}")

        ny, nx = lat2.shape
        if nx < 2 or ny < 2:
            raise RuntimeError("Need at least 2x2 grid to infer projected spacing.")

        # Use GRIB-provided spacing if available
        a = {}
        a.update(getattr(ds, "attrs", {}) or {})
        a.update(var_attrs or {})

        dx = a.get("GRIB_DxInMetres", None)
        dy = a.get("GRIB_DyInMetres", None)
        if dx is None or dy is None:
            # Try projection var attrs
            proj_var = None
            if "gribfile_projection" in ds.coords:
                proj_var = ds.coords["gribfile_projection"]
            elif "gribfile_projection" in ds.variables:
                proj_var = ds["gribfile_projection"]
            if proj_var is not None:
                pa = proj_var.attrs
                dx = dx or pa.get("GRIB_DxInMetres")
                dy = dy or pa.get("GRIB_DyInMetres")

        if dx is None or dy is None:
            raise RuntimeError("Could not find GRIB_DxInMetres/GRIB_DyInMetres needed to synthesize x/y vectors.")

        dx = float(dx)
        dy = float(dy)

        # Determine sign from actual projected neighbors (protects against scan direction flips)
        lon00, lat00 = float(lon2[0, 0]), float(lat2[0, 0])
        lon01, lat01 = float(lon2[0, 1]), float(lat2[0, 1])
        lon10, lat10 = float(lon2[1, 0]), float(lat2[1, 0])

        x00, y00 = self._ll_to_xy.transform(lon00, lat00)
        x01, y01 = self._ll_to_xy.transform(lon01, lat01)
        x10, y10 = self._ll_to_xy.transform(lon10, lat10)

        dx_sign = 1.0 if (x01 - x00) >= 0 else -1.0
        dy_sign = 1.0 if (y10 - y00) >= 0 else -1.0

        x_vec = x00 + np.arange(nx, dtype=np.float64) * (dx_sign * abs(dx))
        y_vec = y00 + np.arange(ny, dtype=np.float64) * (dy_sign * abs(dy))
        return x_vec, y_vec

    def _project_lonlat(self, lon_deg: float, lat_deg: float) -> Tuple[float, float]:
        if self._ll_to_xy is None:
            # Ensure at least one file loaded
            self._ensure_loaded(0)
        xq, yq = self._ll_to_xy.transform(float(lon_deg), float(lat_deg))
        return float(xq), float(yq)

    @staticmethod
    def _bracket_1d(grid: np.ndarray, xq: float, ascending: bool) -> Tuple[int, int]:
        if ascending:
            if xq <= grid[0]:
                return 0, 0
            if xq >= grid[-1]:
                return len(grid) - 1, len(grid) - 1
            i = int(np.searchsorted(grid, xq))
            return i - 1, i
        else:
            if xq >= grid[0]:
                return 0, 0
            if xq <= grid[-1]:
                return len(grid) - 1, len(grid) - 1
            r = grid[::-1]
            i = int(np.searchsorted(r, xq))
            return len(grid) - i - 1, len(grid) - i

    def _xy_bracket_and_weights(self, wf: _WindFile, xq: float, yq: float):
        k0, k1 = self._bracket_1d(wf.x, xq, wf.x_asc)
        j0, j1 = self._bracket_1d(wf.y, yq, wf.y_asc)

        x0 = float(wf.x[k0]); x1 = float(wf.x[k1])
        y0 = float(wf.y[j0]); y1 = float(wf.y[j1])

        tx = 0.0 if x1 == x0 else (xq - x0) / (x1 - x0)
        ty = 0.0 if y1 == y0 else (yq - y0) / (y1 - y0)
        tx = float(np.clip(tx, 0.0, 1.0))
        ty = float(np.clip(ty, 0.0, 1.0))

        w00 = (1 - ty) * (1 - tx)
        w10 = (1 - ty) * tx
        w01 = ty * (1 - tx)
        w11 = ty * tx
        return j0, j1, k0, k1, w00, w10, w01, w11

    def _grid_to_earth(self, u_grid: float, v_grid: float, lon_deg: float, lat_deg: float) -> Tuple[float, float]:
        """Rotate grid-relative (u,v) to earth-relative (east,north) using meridian convergence."""
        key = (round(lat_deg, self.sample_latlon_decimals), round(lon_deg, self.sample_latlon_decimals))
        cached = self._gamma_cache.get(key)
        if cached is None:
            factors = self._proj.get_factors(float(lon_deg), float(lat_deg))
            gamma = float(factors.meridian_convergence)  # degrees
            if self.verbose and not getattr(self, "_printed_gamma_check", False):
                gamma_num = self._gamma_numeric_deg(lon_deg, lat_deg, dx_m=1000.0)
                print(
                    f"[HRRR] gamma check @({lat_deg:.4f},{lon_deg:.4f}): "
                    f"gamma_proj={gamma:.4f} deg, gamma_num={gamma_num:.4f} deg, "
                    f"diff={gamma_num - gamma:+.4f} deg",
                    flush=True
                )
                self._printed_gamma_check = True
            cg = float(np.cos(np.deg2rad(-gamma)))
            sg = float(np.sin(np.deg2rad(-gamma)))
            cached = (cg, sg)
            self._gamma_cache[key] = cached

            if self.verbose and not self._printed_convergence:
                print(f"[HRRR] grid-relative winds detected; meridian_convergence={gamma:.3f} deg "
                    f"at ({lat_deg:.4f},{lon_deg:.4f})", flush=True)
                self._printed_convergence = True

        cg, sg = cached
        u_e = u_grid * cg - v_grid * sg
        v_e = u_grid * sg + v_grid * cg
        return float(u_e), float(v_e)

    def _infer_proj4_from_attrs(self, proj_attrs: dict, ds_attrs: dict) -> str:
        """Fallback if WKT missing. Uses GRIB Lambert parameters."""
        a = {}
        a.update(ds_attrs or {})
        a.update(proj_attrs or {})

        grid_type = str(a.get("GRIB_gridType") or "").lower()
        grid_mapping_name = str(a.get("grid_mapping_name") or "").lower()

        if "lambert" not in grid_type and "lambert" not in grid_mapping_name:
            raise ValueError(f"Unsupported grid for HRRR projected interpolation: {grid_type!r} {grid_mapping_name!r}")

        lat0 = a.get("GRIB_LaDInDegrees") or a.get("latitude_of_projection_origin") or 0.0
        lon0 = a.get("GRIB_LoVInDegrees") or a.get("longitude_of_central_meridian") or 0.0
        lat1 = a.get("GRIB_Latin1InDegrees") or lat0
        lat2 = a.get("GRIB_Latin2InDegrees") or lat1

        return (
            f"+proj=lcc +lat_1={float(lat1)} +lat_2={float(lat2)} "
            f"+lat_0={float(lat0)} +lon_0={float(lon0)} "
            "+a=6371229 +b=6371229 +units=m +no_defs"
        )
