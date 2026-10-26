"""Kennzahlen und Grafik zur Stromerzeugung pro Energieträger (ENTSO-E-Export, Deutschland).

Aufruf:
    python3 kennzahlen_erzeugung.py [pfad/zur/datei.xlsx]

Ohne Argument wird die Datei auf dem Desktop verwendet.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_FILE = Path.home() / "Desktop" / "AGGREGATED_GENERATION_PER_TYPE_GENERATION_202512312300-202612312300.xlsx"
OUTPUT_PNG = Path(__file__).with_name("kennzahlen_erzeugung.png")

# Zeile 6 (Index 5) enthält die Spaltennamen, die Daten beginnen in Zeile 7 (Index 6)
HEADER_ROW = 5
FIRST_DATA_ROW = 6
INTERVAL_HOURS = 0.25  # 15-Minuten-Werte

# Erneuerbare Erzeugung (nach den Spaltennamen der Datei)
RENEWABLE_PREFIXES = (
    "Biomass", "Geothermal", "Hydro Run-of-river", "Hydro Water Reservoir",
    "Marine", "Other renewable", "Solar", "Wind Offshore", "Wind Onshore",
)
# "Energy storage" ist Entladung eines Speichers und wird nicht zur Erzeugung gezählt
EXCLUDED_FROM_GENERATION = ("Energy storage",)


def load_data(path) -> pd.DataFrame:
    # "path" kann ein Dateipfad oder eine hochgeladene Datei sein (z. B. in der Streamlit-App)
    raw = pd.read_excel(path, header=None, engine="openpyxl")
    columns = ["MTU"] + [str(c).replace(" (MW)", "") for c in raw.iloc[HEADER_ROW, 1:]]
    df = raw.iloc[FIRST_DATA_ROW:].copy()
    df.columns = columns

    # Zeitstempel: Startzeit jedes Intervalls. Um die Zeitumstellung (CET/CEST)
    # herum steht zusätzlich "(CET)" bzw. "(CEST)" im Text, das wird hier ignoriert.
    # Im Herbst kommt die Stunde 02:00–03:00 doppelt vor, daher kein eindeutiger Index.
    start = df["MTU"].astype(str).str.split(" - ").str[0]
    stamp = start.str.extract(r"(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})")[0]
    df.index = pd.to_datetime(stamp, format="%d/%m/%Y %H:%M:%S")
    df.index.name = "Zeit"  # sonst heißt der Index "0" und kollidiert im Diagramm
    df = df.drop(columns="MTU")

    # "-" und "n/e" werden zu NaN
    return df.apply(pd.to_numeric, errors="coerce")


def compute_kpis(df: pd.DataFrame) -> dict:
    # Nur Spalten mit mindestens einem Messwert
    df = df.dropna(axis="columns", how="all")
    gen_cols = [c for c in df.columns if c not in EXCLUDED_FROM_GENERATION]
    ren_cols = [c for c in gen_cols if c.startswith(RENEWABLE_PREFIXES)]

    # Nur Zeitpunkte, zu denen alle Erzeugungsspalten Werte haben
    gen = df[gen_cols].dropna()
    total = gen.sum(axis=1)
    renewable = gen[ren_cols].sum(axis=1)
    share = renewable / total

    energy_gwh = gen.sum() * INTERVAL_HOURS / 1000
    stats = pd.DataFrame({
        "Mittelwert [MW]": gen.mean(),
        "Maximum [MW]": gen.max(),
        "Zeitpunkt Maximum": gen.idxmax(),
        "Energie [GWh]": energy_gwh,
    }).sort_values("Energie [GWh]", ascending=False)

    return {
        "period": (gen.index.min(), gen.index.max()),
        "n_intervals": len(gen),
        "gen_cols": gen_cols,
        "ren_cols": ren_cols,
        "skipped_cols": [c for c in EXCLUDED_FROM_GENERATION if c in df.columns],
        "renewable_share_energy": renewable.sum() / total.sum(),
        "renewable_share_mean": share.mean(),
        "renewable_share_min": (share.min(), share.idxmin()),
        "renewable_share_max": (share.max(), share.idxmax()),
        "total_mean_mw": total.mean(),
        "total_max_mw": (total.max(), total.idxmax()),
        "total_energy_twh": total.sum() * INTERVAL_HOURS / 1_000_000,
        "stats": stats,
        "share_daily": share.resample("D").mean(),
    }


def print_report(kpis: dict, all_cols: list[str]) -> None:
    start, end = kpis["period"]
    print("=== Kennzahlen Stromerzeugung Deutschland ===")
    print(f"Zeitraum mit vollständigen Daten: {start:%d.%m.%Y %H:%M} – {end:%d.%m.%Y %H:%M}")
    print(f"Anzahl 15-Minuten-Intervalle: {kpis['n_intervals']}")
    print(f"Ohne Messwerte (übersprungen): {', '.join(c for c in all_cols if c not in kpis['gen_cols'] and c not in EXCLUDED_FROM_GENERATION) or '-'}")
    print()
    print(f"Anteil erneuerbarer Energie (energiegewichtet): {kpis['renewable_share_energy']:.1%}")
    print(f"Anteil erneuerbarer Energie (Mittel über Intervalle): {kpis['renewable_share_mean']:.1%}")
    v, t = kpis["renewable_share_min"]
    print(f"  Minimum: {v:.1%} am {t:%d.%m.%Y %H:%M}")
    v, t = kpis["renewable_share_max"]
    print(f"  Maximum: {v:.1%} am {t:%d.%m.%Y %H:%M}")
    print()
    print(f"Gesamterzeugung Mittelwert: {kpis['total_mean_mw']:,.0f} MW")
    v, t = kpis["total_max_mw"]
    print(f"Gesamterzeugung Maximum: {v:,.0f} MW am {t:%d.%m.%Y %H:%M}")
    print(f"Gesamterzeugung Energie: {kpis['total_energy_twh']:.2f} TWh")
    print()
    print("Pro Energieträger:")
    with pd.option_context("display.float_format", "{:,.1f}".format, "display.width", 120):
        print(kpis["stats"].to_string())


def make_plot(kpis: dict, output: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), constrained_layout=True)

    daily = kpis["share_daily"].dropna() * 100
    ax1.plot(daily.index, daily.values, color="#2a9d8f", linewidth=1.5)
    ax1.set_title("Anteil erneuerbarer Energie (Tagesmittel)")
    ax1.set_ylabel("Anteil [%]")
    ax1.set_ylim(0, 100)
    ax1.grid(alpha=0.3)

    stats = kpis["stats"]
    ax2.barh(stats.index, stats["Energie [GWh]"], color="#457b9d")
    ax2.invert_yaxis()
    ax2.set_title("Erzeugte Energie pro Energieträger")
    ax2.set_xlabel("Energie [GWh]")
    ax2.grid(axis="x", alpha=0.3)

    fig.savefig(output, dpi=150)
    print(f"\nGrafik gespeichert: {output}")


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FILE
    df = load_data(path)
    kpis = compute_kpis(df)
    print_report(kpis, list(df.columns))
    make_plot(kpis, OUTPUT_PNG)


if __name__ == "__main__":
    main()
