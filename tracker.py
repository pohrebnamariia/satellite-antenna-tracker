"""Консольний трекер. Запуск: python tracker.py --help"""
import argparse
from datetime import timedelta

from skyfield.api import load

import tracker_core as core


def main():
    ap = argparse.ArgumentParser(description="Трекер радіоаматорських супутників")
    ap.add_argument("--lat", type=float, default=core.DEFAULT_LAT)
    ap.add_argument("--lon", type=float, default=core.DEFAULT_LON)
    ap.add_argument("--hours", type=int, default=24, help="на скільки годин уперед шукати")
    ap.add_argument("--min-el", type=float, default=20, help="мінімальна максимальна висота, °")
    ap.add_argument("--sats", nargs="+", default=list(core.SATELLITES),
                    help="ключі зі SATELLITES, напр. ISS SO-50")
    ap.add_argument("--doppler", metavar="SAT",
                    help="показати таблицю Доплера для найближчого проліту (напр. SO-50)")
    ap.add_argument("--step", type=int, default=30, help="крок таблиці Доплера, с")
    ap.add_argument("--list-names", action="store_true", help="показати всі назви супутників")
    args = ap.parse_args()

    tz = core.get_local_tz()
    by_name = core.load_satellites()

    if args.list_names:
        print("\n".join(sorted(by_name)))
        return

    observer = core.make_observer(args.lat, args.lon)
    ts = load.timescale()
    now = ts.now()
    end = ts.from_datetime(now.utc_datetime() + timedelta(hours=args.hours))

    selected = []
    for key in args.sats:
        cfg = core.SATELLITES.get(key)
        sat = core.find_satellite(by_name, cfg["match"] if cfg else key)
        if sat is None:
            print(f"Не знайдено: {key}")
        else:
            selected.append((key, sat))

    # ---------- Поточні положення ----------
    print(f"\nПоточне положення ({now.astimezone(tz):%d.%m %H:%M:%S}):")
    print(f"{'SATELLITE':<24}{'AZ':<9}{'EL':<9}{'DIST km':<10}STATUS")
    for key, sat in selected:
        pos = core.current_position(sat, observer, now)
        status = "над горизонтом" if pos["el"] > 0 else "під горизонтом"
        print(f"{sat.name[:23]:<24}{pos['az']:<9.1f}{pos['el']:<9.1f}"
              f"{pos['dist_km']:<10.0f}{status}")

    # ---------- Прольоти ----------
    all_passes = []
    for key, sat in selected:
        for p in core.find_passes(sat, observer, now, end, args.min_el):
            p["key"] = key
            p["tz"] = tz
            all_passes.append(p)
    all_passes.sort(key=lambda p: p["rise"].tt)

    print(f"\nПрольоти з максимумом вище {args.min_el:.0f}° (час Києва):")
    print(f"{'SATELLITE':<24}{'RISE':<13}{'MAX':<10}{'SET':<10}"
          f"{'MAX EL':<8}{'DIST':<7}AZ RISE→SET")
    for p in all_passes:
        print(
            f"{p['name'][:23]:<24}"
            f"{p['rise'].astimezone(tz):%d.%m %H:%M  }"
            f"{p['max'].astimezone(tz):%H:%M:%S  }"
            f"{p['set'].astimezone(tz):%H:%M:%S  }"
            f"{p['max_el']:<8.1f}{p['dist_km']:<7.0f}"
            f"{p['rise_az']:.0f}° → {p['set_az']:.0f}°"
        )
    if not all_passes:
        print("Немає прольотів за заданих умов. Зменш --min-el.")

    # ---------- Доплер ----------
    if args.doppler:
        key = args.doppler
        cfg = core.SATELLITES.get(key)
        if cfg is None:
            print(f"\nДля {key} немає частот у SATELLITES (tracker_core.py).")
            return
        sat = dict(selected).get(key) or core.find_satellite(by_name, cfg["match"])
        next_pass = next((p for p in all_passes if p["key"] == key), None)
        if next_pass is None:
            print(f"\nНемає підходящих прольотів для {key}.")
            return

        print(f"\nДоплер для {sat.name}, проліт "
              f"{next_pass['rise'].astimezone(tz):%d.%m %H:%M:%S} - "
              f"{next_pass['set'].astimezone(tz):%H:%M:%S}, "
              f"максимум {next_pass['max_el']:.1f}°")
        print(f"Номінал: RX {cfg['down']} МГц, TX {cfg['up']} МГц\n")
        print(f"{'TIME':<10}{'AZ':<7}{'EL':<7}{'RX MHz':<12}{'TX MHz':<12}SHIFT RX")
        for r in core.doppler_table(sat, observer, ts, next_pass,
                                    cfg["down"], cfg["up"], args.step):
            print(f"{r['time']:%H:%M:%S}  {r['az']:<7.0f}{r['el']:<7.1f}"
                  f"{r['rx_mhz']:<12.4f}{r['tx_mhz']:<12.4f}{r['shift_khz']:+.2f} kHz")


if __name__ == "__main__":
    main()