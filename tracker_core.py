"""Розрахункове ядро: супутники, прольоти, Доплер, підсупутникова точка."""
from datetime import timedelta, timezone
from pathlib import Path

import numpy as np
from skyfield.api import Loader, wgs84

BASE = Path(__file__).parent
# TLE-файли зберігаються поруч із кодом, незалежно від того, звідки запущено програму
load = Loader(str(BASE))

C_KM_S = 299792.458
DEFAULT_LAT = 50.4497
DEFAULT_LON = 30.4608
ALL_GROUPS = ("amateur", "stations", "weather", "science")

# Типові частоти (МГц). Обов'язково перевір актуальні на amsat.org!
# down = супутник -> ти (RX), up = ти -> супутник (TX). None = немає такої частоти.
SATELLITES = {
    "ISS":     {"match": "ISS (ZARYA)", "down": 437.800,  "up": 145.990, "group": "stations",
                "label": "ISS (ZARYA)", "desc": "Міжнародна космічна станція"},
    "SO-50":   {"match": "SO-50",       "down": 436.795,  "up": 145.850, "group": "amateur",
                "label": "SO-50", "desc": "FM-супутник (SaudiSat 1C)"},
    "RS-44":   {"match": "RS-44",       "down": 435.640,  "up": 145.965, "group": "amateur",
                "label": "RS-44", "desc": "Лінійний транспондер"},
    "FO-29":   {"match": "FO-29",       "down": 435.850,  "up": 145.950, "group": "amateur",
                "label": "FO-29 (JAS-2)", "desc": "Лінійний транспондер"},
    "NOAA-19": {"match": "NOAA 19",     "down": 137.100,  "up": None,    "group": "weather",
                "label": "NOAA-19", "desc": "Метеосупутник, APT-зображення"},
    "NOAA-18": {"match": "NOAA 18",     "down": 137.9125, "up": None,    "group": "weather",
                "label": "NOAA-18", "desc": "Метеосупутник, APT-зображення"},
    "NOAA-15": {"match": "NOAA 15",     "down": 137.620,  "up": None,    "group": "weather",
                "label": "NOAA-15", "desc": "Метеосупутник, APT-зображення"},
    "HST":     {"match": "HST",         "down": None,     "up": None,    "group": "science",
                "label": "Hubble Space Telescope", "desc": "Космічний телескоп"},
}


def get_local_tz():
    try:
        from zoneinfo import ZoneInfo
        try:
            return ZoneInfo("Europe/Kyiv")
        except Exception:
            return ZoneInfo("Europe/Kiev")
    except Exception:
        print("Увага: tzdata не знайдено, беру фіксований UTC+3 (pip install tzdata).")
        return timezone(timedelta(hours=3))


def load_satellites(groups=ALL_GROUPS):
    """Завантажує TLE (не частіше разу на добу) і повертає {назва: супутник}."""
    by_name = {}
    for group in groups:
        filename = f"{group}.tle"
        try:
            if not load.exists(filename) or load.days_old(filename) > 1:
                url = f"https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=tle"
                load.download(url, filename=filename)
            sats = load.tle_file(filename)
        except Exception as e:
            print(f"Не вдалося завантажити групу {group}: {e}")
            continue
        for sat in sats:
            by_name[sat.name] = sat
    return by_name


def find_satellite(by_name, part):
    """Спочатку точний збіг назви, потім пошук за частиною назви."""
    p = part.lower()
    for name, sat in by_name.items():
        if name.lower() == p:
            return sat
    for name, sat in by_name.items():
        if p in name.lower():
            return sat
    return None


def make_observer(lat=DEFAULT_LAT, lon=DEFAULT_LON, elevation_m=0.0):
    return wgs84.latlon(latitude_degrees=lat, longitude_degrees=lon, elevation_m=elevation_m)


def current_position(sat, observer, t):
    """Азимут, кут місця й відстань до супутника в момент t."""
    alt, az, dist = (sat - observer).at(t).altaz()
    return {"az": az.degrees, "el": alt.degrees, "dist_km": dist.km}


def subpoint(sat, t):
    """Точка на Землі під супутником: широта, довгота, висота."""
    sp = wgs84.subpoint(sat.at(t))
    return {"lat": sp.latitude.degrees, "lon": sp.longitude.degrees, "height_km": sp.elevation.km}


def find_passes(sat, observer, t0, t1, min_el=20.0, include_ongoing=True):
    """Прольоти (RISE -> MAX -> SET) з максимумом не нижче min_el.

    Якщо супутник уже над горизонтом у момент t0, поточний проліт теж
    повертається (з прапорцем ongoing=True і справжнім часом RISE у минулому).
    """
    diff = sat - observer
    times, events = sat.find_events(observer, t0, t1, altitude_degrees=0)
    passes, cur = [], {}

    el0, az0, d0 = diff.at(t0).altaz()
    if include_ongoing and el0.degrees > 0:
        tb = t0.ts.tt_jd(t0.tt - 40 / 1440)          # заглядаємо на 40 хв назад
        bt, be = sat.find_events(observer, tb, t0, altitude_degrees=0)
        rises = [t for t, e in zip(bt, be) if e == 0]
        if rises:
            r = rises[-1]
            cur = {"name": sat.name, "rise": r, "ongoing": True,
                   "rise_az": diff.at(r).altaz()[1].degrees}
            maxes = [t for t, e in zip(bt, be) if e == 1 and t.tt > r.tt]
            if maxes:
                alt, _, dist = diff.at(maxes[-1]).altaz()
                cur.update(max=maxes[-1], max_el=alt.degrees, dist_km=dist.km)

    for t, ev in zip(times, events):
        if ev == 0:                                   # RISE
            cur = {"name": sat.name, "rise": t,
                   "rise_az": diff.at(t).altaz()[1].degrees}
        elif ev == 1 and "rise" in cur:               # MAX
            alt, az, dist = diff.at(t).altaz()
            cur.update(max=t, max_el=alt.degrees, dist_km=dist.km)
        elif ev == 2 and "max" in cur:                # SET
            if cur["max_el"] >= min_el:
                cur["set"] = t
                cur["set_az"] = diff.at(t).altaz()[1].degrees
                passes.append(cur)
            cur = {}
    return passes


def sample_pass(sat, observer, ts, p, down_mhz, up_mhz, step_s=10):
    """Масиви азимуту, висоти, відстані та частот по ходу проліту p."""
    diff = sat - observer
    duration = (p["set"].tt - p["rise"].tt) * 86400
    offsets = np.append(np.arange(0, duration, step_s), duration)
    sample = ts.tt_jd(p["rise"].tt + offsets / 86400)

    alt, az, dist = diff.at(sample).altaz()

    # швидкість зміни відстані (км/с) як різниця відстаней на інтервалі в 1 с
    d_before = diff.at(ts.tt_jd(sample.tt - 0.5 / 86400)).distance().km
    d_after = diff.at(ts.tt_jd(sample.tt + 0.5 / 86400)).distance().km
    range_rate = d_after - d_before                   # > 0: віддаляється

    nan = np.full(len(offsets), np.nan)
    rx = down_mhz * (1 - range_rate / C_KM_S) if down_mhz else nan
    tx = up_mhz * (1 + range_rate / C_KM_S) if up_mhz else nan
    shift = (rx - down_mhz) * 1000 if down_mhz else nan

    return {"time": sample, "offsets": offsets, "az": az.degrees, "el": alt.degrees,
            "dist_km": dist.km, "rx_mhz": rx, "tx_mhz": tx, "shift_khz": shift}


def doppler_table(sat, observer, ts, p, down_mhz, up_mhz, step_s=30):
    """Таблиця азимуту, висоти та частот RX/TX по ходу проліту p (для консолі)."""
    s = sample_pass(sat, observer, ts, p, down_mhz, up_mhz, step_s)
    local_times = s["time"].astimezone(p["tz"])
    return [
        {
            "time": local_times[i],
            "az": s["az"][i],
            "el": s["el"][i],
            "dist_km": s["dist_km"][i],
            "rx_mhz": s["rx_mhz"][i],
            "tx_mhz": s["tx_mhz"][i],
            "shift_khz": s["shift_khz"][i],
        }
        for i in range(len(s["offsets"]))
    ]