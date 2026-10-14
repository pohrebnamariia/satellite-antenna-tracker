"""Прості астрономічні розрахунки без зовнішніх файлів: Сонце, схід/захід, фаза Місяця.

Точність: кілька хвилин для сходу/заходу, цього достатньо для інформаційної панелі.
"""
import math
from datetime import datetime

import numpy as np


def _jd(unix):
    return unix / 86400.0 + 2440587.5


def _sun_geometry(jd):
    """Схилення Сонця та довгота підсонячної точки (градуси)."""
    n = jd - 2451545.0
    L = (280.460 + 0.9856474 * n) % 360.0
    g = np.radians((357.528 + 0.9856003 * n) % 360.0)
    lam = np.radians(L + 1.915 * np.sin(g) + 0.020 * np.sin(2 * g))
    eps = np.radians(23.439 - 0.0000004 * n)
    decl = np.arcsin(np.sin(eps) * np.sin(lam))
    ra = np.arctan2(np.cos(eps) * np.sin(lam), np.cos(lam))
    gmst = (18.697374558 + 24.06570982441908 * n) % 24.0
    lon = (np.degrees(ra) - gmst * 15.0 + 180.0) % 360.0 - 180.0
    return np.degrees(decl), lon


def sun_subpoint(dt):
    """Підсонячна точка (широта, довгота) для aware-datetime."""
    decl, lon = _sun_geometry(_jd(dt.timestamp()))
    return float(decl), float(lon)


def sun_altitude(unix, lat, lon):
    decl, slon = _sun_geometry(_jd(unix))
    h = np.radians(lon - slon)
    la = math.radians(lat)
    d = np.radians(decl)
    return np.degrees(np.arcsin(np.sin(la) * np.sin(d) + np.cos(la) * np.cos(d) * np.cos(h)))


def sunrise_sunset(lat, lon, when, tz):
    """Схід і захід Сонця за місцеву добу з моменту when. Повертає (rise, set) або None."""
    start = when.astimezone(tz).replace(hour=0, minute=0, second=0, microsecond=0)
    unix = start.timestamp() + 60.0 * np.arange(0, 1441)
    above = sun_altitude(unix, lat, lon) > -0.833
    up = np.where(~above[:-1] & above[1:])[0]
    down = np.where(above[:-1] & ~above[1:])[0]
    rise = datetime.fromtimestamp(unix[up[0] + 1], tz) if len(up) else None
    set_ = datetime.fromtimestamp(unix[down[0] + 1], tz) if len(down) else None
    return rise, set_


def moon_phase(when):
    syn = 29.530588853
    age = (_jd(when.timestamp()) - 2451550.1) % syn
    f = age / syn
    illum = (1 - math.cos(2 * math.pi * f)) / 2 * 100
    if f < 0.03 or f >= 0.97:
        name = "Новий місяць"
    elif f < 0.22:
        name = "Зростаючий серп"
    elif f < 0.28:
        name = "Перша чверть"
    elif f < 0.47:
        name = "Зростаючий (випуклий)"
    elif f < 0.53:
        name = "Повня"
    elif f < 0.72:
        name = "Спадаючий (випуклий)"
    elif f < 0.78:
        name = "Остання чверть"
    else:
        name = "Спадаючий серп"
    return {"name": name, "illumination": illum, "age_days": age}