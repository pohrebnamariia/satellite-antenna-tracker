"""API для сайту. Запуск: python -m uvicorn api:app --reload"""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from skyfield.api import wgs84

import sky
import tracker_core as core
from tracker_core import load

BASE = Path(__file__).parent
CUSTOM_FILE = BASE / "custom_satellites.json"

app = FastAPI(title="SkyTrack")
ts = load.timescale()
tz = core.get_local_tz()
by_name = core.load_satellites()          # TLE завантажуються один раз при старті


# ---------- власні супутники (кнопка «Додати супутник») ----------
def _register_custom():
    if not CUSTOM_FILE.exists():
        return
    try:
        data = json.loads(CUSTOM_FILE.read_text(encoding="utf-8"))
    except Exception:
        return
    for key, cfg in data.items():
        fn = f"cat_{cfg['catnr']}.tle"
        url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={cfg['catnr']}&FORMAT=TLE"
        try:
            if not load.exists(fn):
                load.download(url, filename=fn)
            sat = load.tle_file(fn)[0]
        except Exception:
            continue
        by_name[sat.name] = sat
        core.SATELLITES[key] = {"match": sat.name, "down": None, "up": None, "group": "custom",
                                "label": sat.name, "desc": f"NORAD {cfg['catnr']}"}


_register_custom()


def get_sat(key: str):
    cfg = core.SATELLITES.get(key)
    if cfg is None:
        raise HTTPException(404, f"Невідомий супутник: {key}")
    sat = core.find_satellite(by_name, cfg["match"])
    if sat is None:
        raise HTTPException(404, f"Супутник не знайдено в TLE: {key}")
    return sat, cfg


def _f(x):
    """numpy -> звичайне число (nan -> None) для JSON."""
    x = float(x)
    return None if np.isnan(x) else x


def _list(a):
    return [_f(v) for v in a]


def _iso(t):
    return t.astimezone(tz).isoformat()


# ---------- список супутників ----------
@app.get("/api/satellites")
def satellites():
    result = []
    for key, cfg in core.SATELLITES.items():
        sat = core.find_satellite(by_name, cfg["match"])
        result.append({
            "key": key,
            "label": cfg.get("label", key),
            "desc": cfg.get("desc", ""),
            "group": cfg.get("group", ""),
            "down": cfg["down"],
            "up": cfg["up"],
            "available": sat is not None,
        })
    return result


@app.post("/api/satellites/add")
def add_satellite(catnr: int):
    """Додає супутник за NORAD-номером (TLE беруться з CelesTrak)."""
    fn = f"cat_{catnr}.tle"
    url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={catnr}&FORMAT=TLE"
    try:
        load.download(url, filename=fn)
        sats = load.tle_file(fn)
    except Exception as e:
        raise HTTPException(502, f"Не вдалося отримати TLE для {catnr}: {e}")
    if not sats:
        raise HTTPException(404, f"CelesTrak не знайшов супутник {catnr}")
    sat = sats[0]
    key = f"N{catnr}"
    by_name[sat.name] = sat
    core.SATELLITES[key] = {"match": sat.name, "down": None, "up": None, "group": "custom",
                            "label": sat.name, "desc": f"NORAD {catnr}"}
    try:
        data = json.loads(CUSTOM_FILE.read_text(encoding="utf-8")) if CUSTOM_FILE.exists() else {}
        data[key] = {"catnr": catnr}
        CUSTOM_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass
    return {"key": key, "label": sat.name}


# ---------- довідкові дані про супутник ----------
@app.get("/api/info")
def info(sat: str = "ISS"):
    s, cfg = get_sat(sat)
    now = ts.now()
    geo = s.at(now)
    sp = wgs84.subpoint(geo)
    m = s.model
    return {
        "key": sat,
        "name": s.name,
        "label": cfg.get("label", sat),
        "desc": cfg.get("desc", ""),
        "norad": int(m.satnum),
        "period_min": float(2 * np.pi / m.no_kozai),
        "inclination_deg": float(np.degrees(m.inclo)),
        "eccentricity": float(m.ecco),
        "altitude_km": float(sp.elevation.km),
        "speed_kms": float(np.linalg.norm(geo.velocity.km_per_s)),
        "tle_epoch": s.epoch.utc_datetime().isoformat(),
        "tle_age_days": float(now.tt - s.epoch.tt),
        "down_mhz": cfg["down"],
        "up_mhz": cfg["up"],
    }


# ---------- поточний стан ----------
@app.get("/api/state")
def state(sat: str = "ISS", lat: float = core.DEFAULT_LAT, lon: float = core.DEFAULT_LON):
    """Положення супутника, відстань, азимут/висота, швидкість і частоти Доплера."""
    s, cfg = get_sat(sat)
    obs = core.make_observer(lat, lon)
    now = ts.now()

    diff = s - obs
    alt, az, dist = diff.at(now).altaz()
    geo = s.at(now)
    sp = wgs84.subpoint(geo)

    d1 = diff.at(ts.tt_jd(now.tt - 0.5 / 86400)).distance().km
    d2 = diff.at(ts.tt_jd(now.tt + 0.5 / 86400)).distance().km
    range_rate = float(d2 - d1)                       # км/с, > 0: віддаляється

    sun_lat, sun_lon = sky.sun_subpoint(now.utc_datetime())
    down, up = cfg["down"], cfg["up"]

    return {
        "name": s.name,
        "time": _iso(now),
        "az": float(az.degrees),
        "el": float(alt.degrees),
        "dist_km": float(dist.km),
        "visible": bool(alt.degrees > 0),
        "sub_lat": float(sp.latitude.degrees),
        "sub_lon": float(sp.longitude.degrees),
        "height_km": float(sp.elevation.km),
        "speed_kms": float(np.linalg.norm(geo.velocity.km_per_s)),
        "range_rate": range_rate,
        "rx_mhz": down * (1 - range_rate / core.C_KM_S) if down else None,
        "tx_mhz": up * (1 + range_rate / core.C_KM_S) if up else None,
        "sun_lat": sun_lat,
        "sun_lon": sun_lon,
    }


# ---------- траєкторія ----------
@app.get("/api/track")
def track(sat: str = "ISS", back_min: int = 20, ahead_min: int = 100, step_s: int = 30):
    """Підсупутникові точки: трохи в минуле й на кількість хвилин уперед."""
    s, _ = get_sat(sat)
    now = ts.now()
    step_s = max(10, step_s)
    offsets = np.arange(-back_min * 60, ahead_min * 60 + 1, step_s)
    t = ts.tt_jd(now.tt + offsets / 86400)
    sp = wgs84.subpoint(s.at(t))
    return {
        "offset_s": offsets.tolist(),
        "lat": _list(sp.latitude.degrees),
        "lon": _list(sp.longitude.degrees),
        "height_km": _list(sp.elevation.km),
    }


# ---------- прольоти ----------
@app.get("/api/passes")
def passes(sat: str = "ISS", lat: float = core.DEFAULT_LAT, lon: float = core.DEFAULT_LON,
           hours: int = 24, min_el: float = 10):
    s, _ = get_sat(sat)
    obs = core.make_observer(lat, lon)
    now = ts.now()
    end = ts.from_datetime(now.utc_datetime() + timedelta(hours=hours))
    result = []
    for p in core.find_passes(s, obs, now, end, min_el):
        result.append({
            "rise": _iso(p["rise"]),
            "max": _iso(p["max"]),
            "set": _iso(p["set"]),
            "max_el": float(p["max_el"]),
            "dist_km": float(p["dist_km"]),
            "rise_az": float(p["rise_az"]),
            "set_az": float(p["set_az"]),
            "ongoing": bool(p.get("ongoing", False)),
        })
    return result


@app.get("/api/profile")
def profile(sat: str = "ISS", lat: float = core.DEFAULT_LAT, lon: float = core.DEFAULT_LON,
            rise: Optional[str] = None, min_el: float = 10, step_s: int = 10):
    """Профіль одного проліту: висота, азимут і частоти кожні step_s секунд."""
    s, cfg = get_sat(sat)
    obs = core.make_observer(lat, lon)
    now = ts.now()
    step_s = max(2, step_s)

    if rise:
        try:
            r = datetime.fromisoformat(rise)
        except ValueError:
            raise HTTPException(400, "Невірний формат часу rise")
        if r.tzinfo is None:
            r = r.replace(tzinfo=timezone.utc)
        t0 = ts.from_datetime(r - timedelta(minutes=2))
        t1 = ts.from_datetime(r + timedelta(minutes=40))
        found = core.find_passes(s, obs, t0, t1, 0, include_ongoing=False)
    else:
        t1 = ts.from_datetime(now.utc_datetime() + timedelta(hours=24))
        found = core.find_passes(s, obs, now, t1, min_el)

    if not found:
        raise HTTPException(404, "Проліт не знайдено")
    p = found[0]
    smp = core.sample_pass(s, obs, ts, p, cfg["down"], cfg["up"], step_s)
    return {
        "rise": _iso(p["rise"]),
        "max": _iso(p["max"]),
        "set": _iso(p["set"]),
        "max_el": float(p["max_el"]),
        "ongoing": bool(p.get("ongoing", False)),
        "offset_s": _list(smp["offsets"]),
        "el": _list(smp["el"]),
        "az": _list(smp["az"]),
        "dist_km": _list(smp["dist_km"]),
        "rx_mhz": _list(smp["rx_mhz"]),
        "tx_mhz": _list(smp["tx_mhz"]),
        "shift_khz": _list(smp["shift_khz"]),
    }


# ---------- Сонце, Місяць ----------
@app.get("/api/quick")
def quick(lat: float = core.DEFAULT_LAT, lon: float = core.DEFAULT_LON):
    now = datetime.now(tz)
    rise, set_ = sky.sunrise_sunset(lat, lon, now, tz)
    return {
        "sunrise": rise.isoformat() if rise else None,
        "sunset": set_.isoformat() if set_ else None,
        "moon": sky.moon_phase(now),
    }


app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")