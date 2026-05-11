"""
ExoTransit Planner — FastAPI backend
Wraps TransitEphemeris to serve transit data + airmass plot traces as JSON.
"""

import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ephem
import numpy as np
import pandas as pd
import pytz
from astropy.coordinates import SkyCoord, get_body_barycentric
from astropy.time import Time
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse

# ── Data loading ──────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent


def merge_swarthmore_ephemeris(nasa_df: pd.DataFrame) -> pd.DataFrame:
    try:
        sw = pd.read_csv(
            "https://astro.swarthmore.edu/transits/transit_targets.csv", comment="#"
        )
        sw["pl_name"] = sw["name"].str.strip()
        sw = sw.rename(columns={"epoch": "sw_tranmid", "period": "sw_period"})[
            ["pl_name", "sw_tranmid", "sw_period"]
        ]
    except Exception as e:
        warnings.warn(f"Could not fetch Swarthmore ephemeris: {e}")
        return nasa_df

    if "default_flag" in nasa_df.columns:
        defaults = nasa_df[nasa_df["default_flag"] == 1].drop_duplicates(
            subset="pl_name", keep="first"
        )
        fallback = nasa_df[
            ~nasa_df["pl_name"].isin(defaults["pl_name"])
        ].drop_duplicates(subset="pl_name", keep="first")
        nasa_dedup = pd.concat([defaults, fallback]).reset_index(drop=True)
    else:
        nasa_dedup = nasa_df.drop_duplicates(
            subset="pl_name", keep="first"
        ).reset_index(drop=True)

    merged = nasa_dedup.merge(sw, on="pl_name", how="left")
    has_sw = merged["sw_tranmid"].notna()
    merged.loc[has_sw, "pl_tranmid"] = merged.loc[has_sw, "sw_tranmid"]
    merged.loc[has_sw, "pl_orbper"] = merged.loc[has_sw, "sw_period"]
    merged.loc[has_sw, "pl_tsystemref"] = "BJD-TDB"
    merged = merged.drop(columns=["sw_tranmid", "sw_period"])
    return merged


def load_data() -> pd.DataFrame:
    csv_files = sorted(DATA_DIR.glob("TD_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No TD_*.csv found in {DATA_DIR}")
    df = pd.read_csv(csv_files[0], delimiter=",", header=0, comment="#")
    df_obs = df.loc[(df.dec > -70) & (df.dec < 20)]
    df_obs = df_obs.loc[~pd.isna(df_obs["pl_tsystemref"])]
    return merge_swarthmore_ephemeris(df_obs)


# ── TransitEphemeris (adapted from final_exo.py) ─────────────────────


class TransitEphemeris:
    BASELINE_PAD_HRS = 1.5

    SITES = {
        "obstech": {
            "lat": "-30.470492",
            "lon": "-70.765483",
            "elev": 1580.0,
            "tz": "America/Santiago",
            "label": "El Sauce Observatory",
            "coords_label": "30°28'S 70°45'W",
            "elev_label": "1580m",
        },
        "keck": {
            "lat": "19.8264",
            "lon": "-155.4744",
            "elev": 4145.0,
            "tz": "Pacific/Honolulu",
            "label": "W. M. Keck Observatory",
            "coords_label": "19°49'N 155°28'W",
            "elev_label": "4145m",
        },
    }

    def __init__(self, df: pd.DataFrame, site: str = "obstech"):
        self._df = df.copy()
        self._site = site
        self._site_cfg = self.SITES[site]
        self._local_tz = pytz.timezone(self._site_cfg["tz"])
        self._t0s = {}
        self._observer = self._make_observer()
        self._precompute_t0s()

    def _with_site(self, site: str):
        """Return a lightweight copy configured for a different site."""
        if site == self._site:
            return self
        clone = object.__new__(type(self))
        clone._df = self._df
        clone._t0s = self._t0s  # ephemerides are site-independent
        clone._site = site
        clone._site_cfg = self.SITES[site]
        clone._local_tz = pytz.timezone(clone._site_cfg["tz"])
        clone._observer = clone._make_observer()
        clone.BASELINE_PAD_HRS = self.BASELINE_PAD_HRS
        return clone

    def _make_observer(self, horizon="-18") -> ephem.Observer:
        obs = ephem.Observer()
        obs.lat = self._site_cfg["lat"]
        obs.lon = self._site_cfg["lon"]
        obs.elevation = self._site_cfg["elev"]
        obs.pressure = 0
        obs.horizon = horizon
        return obs

    def _get_twilight_window(self, mid_utc_dt) -> tuple:
        obs = self._make_observer()
        obs.date = ephem.Date(mid_utc_dt.strftime("%Y/%m/%d %H:%M:%S"))
        sun = ephem.Sun()
        prev_dusk = (
            ephem.Date(obs.previous_setting(sun, use_center=True))
            .datetime()
            .replace(tzinfo=timezone.utc)
        )
        obs.date = ephem.Date(mid_utc_dt.strftime("%Y/%m/%d %H:%M:%S"))
        next_dusk = (
            ephem.Date(obs.next_setting(sun, use_center=True))
            .datetime()
            .replace(tzinfo=timezone.utc)
        )
        if abs((mid_utc_dt - prev_dusk).total_seconds()) <= abs(
            (next_dusk - mid_utc_dt).total_seconds()
        ):
            dusk_utc = prev_dusk
        else:
            dusk_utc = next_dusk
        obs.date = ephem.Date(dusk_utc.strftime("%Y/%m/%d %H:%M:%S"))
        dawn_utc = (
            ephem.Date(obs.next_rising(sun, use_center=True))
            .datetime()
            .replace(tzinfo=timezone.utc)
        )
        return dusk_utc, dawn_utc

    def _to_bjd_tdb(self, jd, time_system, ra, dec):
        ts = time_system.strip().upper() if isinstance(time_system, str) else ""
        if ts in ("BJD", "BJD-TDB", "BJD_TDB", "BJDTDB", "BJD-TBD"):
            return Time(jd, format="jd", scale="tdb")
        elif ts in ("BJD-UTC", "BJD_UTC"):
            return Time(jd, format="jd", scale="utc").tdb
        elif ts in ("HJD", "HJD-TDB", "HJD_TDB", "HJDTDB"):
            from astropy import units as u

            t = Time(jd, format="jd", scale="tdb")
            target = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
            sun_bary = get_body_barycentric("sun", t)
            n_hat = target.cartesian.xyz.value
            r_sun = sun_bary.xyz.to(u.lightsecond).value
            delta = np.dot(r_sun, n_hat) / 86400.0
            return Time(jd + delta, format="jd", scale="tdb")
        elif ts in ("HJD-UTC", "HJD_UTC"):
            from astropy import units as u

            t = Time(jd, format="jd", scale="utc")
            target = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
            sun_bary = get_body_barycentric("sun", t)
            n_hat = target.cartesian.xyz.value
            r_sun = sun_bary.xyz.to(u.lightsecond).value
            delta = np.dot(r_sun, n_hat) / 86400.0
            return Time(jd + delta, format="jd", scale="utc").tdb
        elif ts == "JD":
            return Time(jd, format="jd", scale="utc").tdb
        else:
            return Time(jd, format="jd", scale="tdb")

    def _precompute_t0s(self):
        if "default_flag" in self._df.columns:
            df_default = self._df[self._df["default_flag"] == 1].drop_duplicates(
                subset="pl_name", keep="first"
            )
            df_fallback = self._df[
                ~self._df["pl_name"].isin(df_default["pl_name"])
            ].drop_duplicates(subset="pl_name", keep="first")
            df_to_use = pd.concat([df_default, df_fallback])
        else:
            df_to_use = self._df.drop_duplicates(subset="pl_name", keep="first")

        for _, row in df_to_use.iterrows():
            if pd.isna(row["pl_tranmid"]) or pd.isna(row["pl_orbper"]):
                continue
            try:
                t0 = self._to_bjd_tdb(
                    row["pl_tranmid"], row.get("pl_tsystemref"), row["ra"], row["dec"]
                )
                self._t0s[row["pl_name"]] = t0
            except Exception as e:
                warnings.warn(f"Could not convert t0 for {row['pl_name']}: {e}")

    def _fmt(self, jd):
        t = Time(jd, format="jd", scale="tdb")
        utc_dt = t.utc.to_datetime(timezone.utc)
        local_dt = utc_dt.astimezone(self._local_tz)
        return {
            "utc": utc_dt.strftime("%Y-%m-%d %H:%M UTC"),
            "local": local_dt.strftime("%Y-%m-%d %H:%M %Z"),
            "utc_dt": utc_dt,
            "local_dt": local_dt,
        }

    def _compute_window(self, mid_jd, duration_hrs):
        pad = self.BASELINE_PAD_HRS / 24.0
        half = (duration_hrs / 2.0) / 24.0
        return {
            "baseline_start": self._fmt(mid_jd - half - pad),
            "transit_start": self._fmt(mid_jd - half),
            "transit_mid": self._fmt(mid_jd),
            "transit_end": self._fmt(mid_jd + half),
            "baseline_end": self._fmt(mid_jd + half + pad),
            "duration_hrs": duration_hrs,
        }

    def _moon_above_during_transit(self, transit_start_dt, transit_end_dt):
        obs = self._make_observer(horizon="0")
        moon = ephem.Moon()
        for t in [transit_start_dt, transit_end_dt]:
            obs.date = ephem.Date(t.strftime("%Y/%m/%d %H:%M:%S"))
            moon.compute(obs)
            if moon.alt < 0:
                return False
        return True

    def _moon_info(
        self, transit_start_dt, transit_mid_dt, transit_end_dt, ra_deg, dec_deg
    ):
        obs = self._make_observer()
        moon = ephem.Moon()
        target = ephem.FixedBody()
        target._ra = ephem.degrees(np.radians(ra_deg))
        target._dec = ephem.degrees(np.radians(dec_deg))

        def moon_alt_at(dt):
            obs.date = ephem.Date(dt.strftime("%Y/%m/%d %H:%M:%S"))
            moon.compute(obs)
            return float(np.degrees(moon.alt))

        alt_start = moon_alt_at(transit_start_dt)
        alt_mid = moon_alt_at(transit_mid_dt)
        alt_end = moon_alt_at(transit_end_dt)

        obs.date = ephem.Date(transit_mid_dt.strftime("%Y/%m/%d %H:%M:%S"))
        moon.compute(obs)
        target.compute(obs)
        separation_deg = float(np.degrees(ephem.separation(moon, target)))
        phase_pct = float(moon.phase)

        return {
            "moon_alt_start_deg": round(alt_start, 1),
            "moon_alt_mid_deg": round(alt_mid, 1),
            "moon_alt_end_deg": round(alt_end, 1),
            "moon_sep_deg": round(separation_deg, 1),
            "moon_phase_pct": round(phase_pct, 1),
        }

    def _check_target_alt(
        self,
        baseline_start_dt,
        transit_start_dt,
        transit_end_dt,
        baseline_end_dt,
        ra_deg,
        dec_deg,
        min_alt_deg=33.0,
        cadence_minutes=5,
    ):
        obs = self._make_observer(horizon="0")
        target = ephem.FixedBody()
        target._ra = ephem.degrees(np.radians(ra_deg))
        target._dec = ephem.degrees(np.radians(dec_deg))
        dt = timedelta(minutes=cadence_minutes)

        def sample_window(start_dt, end_dt):
            n_total, n_below = 0, 0
            t = start_dt
            while t <= end_dt:
                obs.date = ephem.Date(t.strftime("%Y/%m/%d %H:%M:%S"))
                target.compute(obs)
                if float(np.degrees(target.alt)) < min_alt_deg:
                    n_below += 1
                n_total += 1
                t += dt
            return n_total, n_below

        n_total_transit, n_below_transit = sample_window(
            transit_start_dt, transit_end_dt
        )
        transit_visible = n_below_transit == 0
        n_total_pre, n_below_pre = sample_window(baseline_start_dt, transit_start_dt)
        n_total_post, n_below_post = sample_window(transit_end_dt, baseline_end_dt)

        pre_pct_lost = (
            round(100.0 * n_below_pre / n_total_pre, 1) if n_total_pre > 0 else 0.0
        )
        post_pct_lost = (
            round(100.0 * n_below_post / n_total_post, 1) if n_total_post > 0 else 0.0
        )

        return {
            "transit_visible": transit_visible,
            "baseline_flag": (pre_pct_lost > 0) or (post_pct_lost > 0),
            "pre_baseline_pct_lost": pre_pct_lost,
            "post_baseline_pct_lost": post_pct_lost,
        }

    @property
    def planets(self):
        return list(self._t0s.keys())

    def get_transits(
        self, planet_name, t_start, t_end, night=False, avoid_darktime=True
    ):
        if planet_name not in self._t0s:
            raise KeyError(f"No valid ephemeris for '{planet_name}'.")

        candidates = self._df[self._df["pl_name"] == planet_name]
        if (
            "default_flag" in candidates.columns
            and (candidates["default_flag"] == 1).any()
        ):
            row = candidates[candidates["default_flag"] == 1].iloc[0]
        else:
            row = candidates.iloc[0]

        t0 = self._t0s[planet_name]
        period = row["pl_orbper"]
        duration = row["pl_trandur"]
        if pd.isna(duration):
            duration = 0.0

        n_start = int(np.floor((t_start.tdb.jd - t0.jd) / period))
        n_end = int(np.ceil((t_end.tdb.jd - t0.jd) / period))

        windows = []
        for n in range(n_start, n_end + 1):
            mid_jd = t0.jd + n * period
            mid_utc = Time(mid_jd, format="jd", scale="tdb").utc
            if not (t_start.utc <= mid_utc <= t_end.utc):
                continue

            window = self._compute_window(mid_jd, duration)
            window["moon"] = self._moon_info(
                window["transit_start"]["utc_dt"],
                window["transit_mid"]["utc_dt"],
                window["transit_end"]["utc_dt"],
                row["ra"],
                row["dec"],
            )

            if night:
                mid_utc_dt = window["transit_mid"]["utc_dt"]
                dusk_utc, dawn_utc = self._get_twilight_window(mid_utc_dt)
                transit_start_dt = window["transit_start"]["utc_dt"]
                transit_end_dt = window["transit_end"]["utc_dt"]
                baseline_start_dt = window["baseline_start"]["utc_dt"]
                baseline_end_dt = window["baseline_end"]["utc_dt"]

                if not (transit_start_dt >= dusk_utc and transit_end_dt <= dawn_utc):
                    continue

                pre_total = (transit_start_dt - baseline_start_dt).total_seconds()
                post_total = (baseline_end_dt - transit_end_dt).total_seconds()
                pre_lost = max(0, (dusk_utc - baseline_start_dt).total_seconds())
                post_lost = max(0, (baseline_end_dt - dawn_utc).total_seconds())

                pre_pct = (
                    round(100.0 * min(pre_lost, pre_total) / pre_total, 1)
                    if pre_total > 0
                    else 0.0
                )
                post_pct = (
                    round(100.0 * min(post_lost, post_total) / post_total, 1)
                    if post_total > 0
                    else 0.0
                )

                window["night_check"] = {
                    "dusk_utc": dusk_utc.isoformat(),
                    "dawn_utc": dawn_utc.isoformat(),
                    "twilight_flag": (pre_pct > 0) or (post_pct > 0),
                    "pre_baseline_pct_lost": pre_pct,
                    "post_baseline_pct_lost": post_pct,
                }

            if avoid_darktime:
                if not self._moon_above_during_transit(
                    window["transit_start"]["utc_dt"], window["transit_end"]["utc_dt"]
                ):
                    continue

            alt_check = self._check_target_alt(
                window["baseline_start"]["utc_dt"],
                window["transit_start"]["utc_dt"],
                window["transit_end"]["utc_dt"],
                window["baseline_end"]["utc_dt"],
                row["ra"],
                row["dec"],
            )
            if not alt_check["transit_visible"]:
                continue

            window["alt_check"] = alt_check
            windows.append(window)

        return windows

    def compute_airmass_traces(self, results: dict, obs_date: str) -> dict:
        """
        Compute all airmass plot data as JSON-serializable dicts for Plotly.
        Returns times, moon trace, target traces with baseline/transit segments,
        twilight boundaries, etc.
        """
        obs = self._make_observer(horizon="0")

        first_windows = next(iter(results.values()))
        mid_utc_dt = first_windows[0]["transit_mid"]["utc_dt"]
        dusk_utc, dawn_utc = self._get_twilight_window(mid_utc_dt)

        plot_start = dusk_utc - timedelta(hours=0.5)
        plot_end = dawn_utc + timedelta(hours=0.5)

        n_steps = int((plot_end - plot_start).total_seconds() / 300) + 1
        times_utc = [plot_start + timedelta(minutes=5 * i) for i in range(n_steps)]
        times_local = [t.astimezone(self._local_tz) for t in times_utc]
        times_iso = [t.isoformat() for t in times_local]

        def get_alt(body):
            alts = []
            for t in times_utc:
                obs.date = ephem.Date(t.strftime("%Y/%m/%d %H:%M:%S"))
                body.compute(obs)
                alts.append(round(float(np.degrees(body.alt)), 2))
            return alts

        # Moon
        moon_alts = get_alt(ephem.Moon())

        # Targets
        targets = []
        for planet_name, windows in results.items():
            row = self._df[self._df["pl_name"] == planet_name].iloc[0]
            target = ephem.FixedBody()
            target._ra = ephem.degrees(np.radians(row["ra"]))
            target._dec = ephem.degrees(np.radians(row["dec"]))
            alts = get_alt(target)

            # Peak for label placement
            peak_idx = int(np.argmax(alts))

            # Segment masks for each window
            segments = []
            for w in windows:
                bl_start = w["baseline_start"]["local_dt"]
                bl_end = w["baseline_end"]["local_dt"]
                t_start = w["transit_start"]["local_dt"]
                t_end = w["transit_end"]["local_dt"]

                bl_mask = [bl_start <= t <= bl_end for t in times_local]
                tr_mask = [t_start <= t <= t_end for t in times_local]

                segments.append(
                    {
                        "baseline_indices": [i for i, m in enumerate(bl_mask) if m],
                        "transit_indices": [i for i, m in enumerate(tr_mask) if m],
                        "transit_start_iso": t_start.isoformat(),
                        "transit_end_iso": t_end.isoformat(),
                    }
                )

            targets.append(
                {
                    "name": planet_name,
                    "alts": alts,
                    "peak_idx": peak_idx,
                    "segments": segments,
                }
            )

        return {
            "times": times_iso,
            "dusk": dusk_utc.astimezone(self._local_tz).isoformat(),
            "dawn": dawn_utc.astimezone(self._local_tz).isoformat(),
            "moon_alts": moon_alts,
            "targets": targets,
        }

    def get_viable_transits_json(self, obs_date: str, bright_only: bool = True):
        """
        Run the full transit search and return JSON-safe results.
        """
        noon_start = self._local_tz.localize(
            datetime.strptime(obs_date + " 12:00:00", "%Y-%m-%d %H:%M:%S")
        )
        noon_end = noon_start + timedelta(hours=24)

        t_start = Time(
            noon_start.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            scale="utc",
        )
        t_end = Time(
            noon_end.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            scale="utc",
        )

        results = {}
        for planet_name in self.planets:
            row = self._df[self._df["pl_name"] == planet_name].iloc[0]
            if pd.isna(row["pl_trandur"]):
                continue
            try:
                windows = self.get_transits(
                    planet_name,
                    t_start=t_start,
                    t_end=t_end,
                    night=True,
                    avoid_darktime=bright_only,
                )
            except Exception:
                continue
            if len(windows) > 0:
                results[planet_name] = windows

        if not results:
            return {
                "table": [], "plot": None, "obs_date": obs_date, "count": 0,
                "site": self._site_cfg,
                "tz_name": self._site_cfg["tz"],
            }

        # Build table rows
        table = []
        for planet_name, windows in results.items():
            candidates = self._df[self._df["pl_name"] == planet_name]
            if (
                "default_flag" in candidates.columns
                and (candidates["default_flag"] == 1).any()
            ):
                row = candidates[candidates["default_flag"] == 1].iloc[0]
            else:
                row = candidates.iloc[0]

            for w in windows:

                def _safe(val, dec=2):
                    if pd.isna(val):
                        return None
                    return round(float(val), dec)

                table.append(
                    {
                        "planet": planet_name,
                        "period_d": _safe(row.get("pl_orbper"), 4),
                        "duration_hr": _safe(row.get("pl_trandur"), 2),
                        "depth_pct": _safe(row.get("pl_trandep"), 2),
                        "r_jup": _safe(row.get("pl_radj"), 2),
                        "r_earth": _safe(row.get("pl_rade"), 2),
                        "vmag": _safe(row.get("sy_vmag"), 1),
                        "ra": _safe(row.get("ra"), 4),
                        "dec": _safe(row.get("dec"), 4),
                        "baseline_start_utc": w["baseline_start"]["utc"],
                        "baseline_start_local": w["baseline_start"]["local"],
                        "transit_start_utc": w["transit_start"]["utc"],
                        "transit_start_local": w["transit_start"]["local"],
                        "transit_mid_utc": w["transit_mid"]["utc"],
                        "transit_mid_local": w["transit_mid"]["local"],
                        "transit_end_utc": w["transit_end"]["utc"],
                        "transit_end_local": w["transit_end"]["local"],
                        "baseline_end_utc": w["baseline_end"]["utc"],
                        "baseline_end_local": w["baseline_end"]["local"],
                        "moon_alt_start": w["moon"]["moon_alt_start_deg"],
                        "moon_alt_mid": w["moon"]["moon_alt_mid_deg"],
                        "moon_alt_end": w["moon"]["moon_alt_end_deg"],
                        "moon_sep": w["moon"]["moon_sep_deg"],
                        "moon_phase": w["moon"]["moon_phase_pct"],
                        "transit_visible": w["alt_check"]["transit_visible"],
                        "baseline_flag": w["alt_check"]["baseline_flag"],
                        "pre_bl_lost": w["alt_check"]["pre_baseline_pct_lost"],
                        "post_bl_lost": w["alt_check"]["post_baseline_pct_lost"],
                        "twilight_flag": w.get("night_check", {}).get(
                            "twilight_flag", False
                        ),
                        "tw_pre_lost": w.get("night_check", {}).get(
                            "pre_baseline_pct_lost", 0
                        ),
                        "tw_post_lost": w.get("night_check", {}).get(
                            "post_baseline_pct_lost", 0
                        ),
                    }
                )

        # Build plot traces
        plot_data = self.compute_airmass_traces(results, obs_date)

        return {
            "table": table,
            "plot": plot_data,
            "obs_date": obs_date,
            "count": len(results),
            "site": self._site_cfg,
            "tz_name": self._site_cfg["tz"],
        }


# ── FastAPI app ───────────────────────────────────────────────────────

app = FastAPI(title="ExoTransit Planner")

# Global ephemeris instance
eph: TransitEphemeris | None = None


@app.on_event("startup")
def startup():
    global eph
    print("Loading exoplanet data and computing ephemerides...")
    df = load_data()
    eph = TransitEphemeris(df)
    print(f"Ready — {len(eph.planets)} planets with valid ephemerides.")


@app.get("/api/transits")
def get_transits(
    obs_date: str = Query(..., description="Observing date YYYY-MM-DD"),
    bright_only: bool = Query(True, description="Only show bright-time transits"),
    site: str = Query("obstech", description="Observatory site: obstech or keck"),
):
    try:
        datetime.strptime(obs_date, "%Y-%m-%d")
    except ValueError:
        return JSONResponse(
            {"error": "Invalid date format. Use YYYY-MM-DD."}, status_code=400
        )

    if site not in TransitEphemeris.SITES:
        return JSONResponse(
            {"error": f"Unknown site '{site}'. Use: {list(TransitEphemeris.SITES.keys())}"},
            status_code=400,
        )

    site_eph = eph._with_site(site)
    result = site_eph.get_viable_transits_json(obs_date, bright_only=bright_only)
    return result


@app.get("/")
def serve_index():
    return FileResponse(DATA_DIR / "index.html")
