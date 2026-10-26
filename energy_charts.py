"""Holt Erzeugungs- und Preisdaten automatisch aus dem Internet.

Quelle: Energy-Charts vom Fraunhofer ISE. Die Daten sind öffentlich, es braucht keinen Schlüssel.
Die Strompreise stammen von der Bundesnetzagentur (SMARD), Lizenz CC BY 4.0.
"""

import json
import ssl
import time
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request
from datetime import date

import certifi
import pandas as pd

API = "https://api.energy-charts.info"
# Zertifikate mitschicken, damit die sichere Verbindung auch auf dem Mac funktioniert
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
TIMEZONE = "Europe/Berlin"
PRICE_COLUMN = "Preis [EUR/MWh]"
INTERVAL_HOURS = 0.25  # 15 Minuten

# Name bei Energy-Charts -> Spaltenname wie in der Excel-Datei
NAME_MAP = {
    "Biomass": "Biomass",
    "Fossil brown coal / lignite": "Fossil Brown coal/Lignite",
    "Fossil hard coal": "Fossil Hard coal",
    "Fossil oil": "Fossil Oil",
    "Fossil coal-derived gas": "Fossil Coal-derived gas",
    "Fossil gas": "Fossil Gas",
    "Geothermal": "Geothermal",
    "Hydro Run-of-River": "Hydro Run-of-river and pondage",
    "Hydro water reservoir": "Hydro Water Reservoir",
    "Hydro pumped storage": "Hydro Pumped Storage",
    "Others": "Other",
    "Waste": "Waste",
    "Wind offshore": "Wind Offshore",
    "Wind onshore": "Wind Onshore",
    "Solar": "Solar",
}


def _get_json(endpoint: str, params: dict) -> dict:
    """Fragt die Schnittstelle ab und gibt die Antwort als Python-Daten zurück."""
    url = f"{API}/{endpoint}?{urllib.parse.urlencode(params)}"
    # Bei "zu viele Anfragen" (Fehler 429) wartet die App kurz und versucht es erneut
    waits = [5, 10, 20]
    for attempt in range(len(waits) + 1):
        try:
            with urllib.request.urlopen(url, timeout=30, context=SSL_CONTEXT) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == len(waits):
                raise
            time.sleep(waits[attempt])


def _to_local_time(unix_seconds: list) -> pd.DatetimeIndex:
    """Wandelt Unix-Sekunden (Zeit seit 1970) in deutsche Uhrzeiten um."""
    return pd.to_datetime(unix_seconds, unit="s", utc=True).tz_convert(TIMEZONE).tz_localize(None)


def fetch_generation(start: date, end: date) -> pd.DataFrame:
    """Erzeugung in MW pro Energieträger in 15-Minuten-Schritten. Das Enddatum ist nicht enthalten."""
    # Die Schnittstelle erwartet die Zeiten in UTC (Weltzeit), deshalb rechnen wir um
    start_utc = pd.Timestamp(start, tz=TIMEZONE).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M")
    end_utc = pd.Timestamp(end, tz=TIMEZONE).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M")
    data = _get_json("public_power", {
        "country": "de",
        "time_step": "15_minutes",
        "start": start_utc,
        "end": end_utc,
    })
    columns = {
        NAME_MAP[item["name"]]: item["data"]
        for item in data["production_types"]
        if item["name"] in NAME_MAP
    }
    df = pd.DataFrame(columns, index=_to_local_time(data["unix_seconds"]))
    df.index.name = "Zeit"
    # Bei der Zeitumstellung im Herbst kommt eine Stunde doppelt vor. Wir behalten das erste Vorkommen,
    # sonst lassen sich Erzeugung und Preise nicht zusammenführen.
    df = df[~df.index.duplicated(keep="first")]
    return df[df.index < pd.Timestamp(end)]


def fetch_prices(start: date, end: date) -> pd.Series:
    """Börsenstrompreis (Day-Ahead) in Euro pro Megawattstunde. Das Enddatum ist nicht enthalten."""
    data = _get_json("price", {
        "bzn": "DE-LU",
        "start": int(pd.Timestamp(start, tz=TIMEZONE).timestamp()),
        "end": int(pd.Timestamp(end, tz=TIMEZONE).timestamp()),
    })
    prices = pd.Series(data["price"], index=_to_local_time(data["unix_seconds"]), name=PRICE_COLUMN)
    # Ältere Preise sind stündlich, neuere viertelstündlich. Wir machen alles viertelstündlich,
    # indem jeder Preis für seine Viertelstunden gilt. Sonst fallen beim Verbinden mit der Erzeugung Werte weg.
    # Doppelte Zeitpunkte (Zeitumstellung im Herbst) zuerst entfernen, sonst klappt das Umrechnen nicht
    prices = prices[~prices.index.duplicated(keep="first")]
    prices = prices.resample("15min").ffill()
    prices.index.name = "Zeit"
    return prices[prices.index < pd.Timestamp(end)]


def compute_revenue(generation: pd.DataFrame, prices: pd.Series) -> pd.DataFrame:
    """Erlös pro Energieträger: erzeugte Menge mal Preis zu jedem Zeitpunkt.

    Hinweis: Das ist der reine Markterlös. Förderungen, Verträge und Kosten sind nicht enthalten.
    """
    # Speicherentladung ist keine Erzeugung, leere Spalten werden ignoriert
    sources = generation.drop(columns=["Energy storage"], errors="ignore").dropna(axis=1, how="all")
    joined = sources.join(prices, how="inner").dropna()
    price = joined[PRICE_COLUMN]
    energy_mwh = joined.drop(columns=PRICE_COLUMN) * INTERVAL_HOURS

    result = pd.DataFrame({
        "Energie [GWh]": energy_mwh.sum() / 1000,
        "Erlös [Mio. €]": energy_mwh.mul(price, axis=0).sum() / 1_000_000,
    })
    # Durchschnittlich erzielter Preis: Erlös geteilt durch Menge
    result["Erzielter Preis [€/MWh]"] = (
        result["Erlös [Mio. €]"] * 1000 / result["Energie [GWh]"]
    ).where(result["Energie [GWh]"] > 0)
    return result.sort_values("Erlös [Mio. €]", ascending=False)


def baseline_last_week(actual: pd.Series) -> pd.Series:
    """Einfacher Vergleichswert: Die Prognose für eine Viertelstunde ist der Wert von vor sieben Tagen."""
    return actual.shift(7 * 96).rename("Prognose [MW]")  # 96 Viertelstunden pro Tag


def read_forecast_csv(file) -> pd.Series:
    """Liest eine eigene Prognose: erste Spalte Zeit, zweite Spalte Megawatt."""
    raw = pd.read_csv(file, sep=None, engine="python")
    forecast = pd.Series(raw.iloc[:, 1].values, index=pd.to_datetime(raw.iloc[:, 0]), name="Prognose [MW]")
    return forecast.sort_index()


def compare_forecast(actual: pd.Series, forecast: pd.Series) -> tuple[pd.DataFrame, dict]:
    """Vergleicht Prognose und echte Werte. Gibt die gemeinsamen Zeitpunkte und die Kennzahlen zurück."""
    joined = pd.concat([actual.rename("Ist [MW]"), forecast], axis=1).dropna()
    error = joined["Prognose [MW]"] - joined["Ist [MW]"]
    metrics = {
        "anzahl": len(joined),
        # Abweichung in Prozent der Gesamtmenge: alle Fehler (ohne Vorzeichen) geteilt durch die echte Menge
        "abweichung_prozent": error.abs().sum() / joined["Ist [MW]"].sum() * 100 if len(joined) else float("nan"),
        # Mittlere Abweichung in MW pro Viertelstunde, ohne Vorzeichen
        "mittlere_abweichung_mw": error.abs().mean() if len(joined) else float("nan"),
        # Systematische Richtung: positiv heißt, die Prognose war im Schnitt zu hoch
        "richtung_mw": error.mean() if len(joined) else float("nan"),
    }
    return joined, metrics


def compute_price_risk(generation: pd.DataFrame, prices: pd.Series) -> pd.DataFrame:
    """Preisrisiko pro Energieträger.

    Erlösfaktor: erzielter Preis im Verhältnis zum Durchschnitt aller Quellen (100 % = Durchschnitt).
    Negativer Anteil: Anteil der Erzeugung, der zu negativen Preisen stattfand.
    """
    sources = generation.drop(columns=["Energy storage"], errors="ignore").dropna(axis=1, how="all")
    joined = sources.join(prices, how="inner").dropna()
    price = joined[PRICE_COLUMN]
    energy = joined.drop(columns=PRICE_COLUMN) * INTERVAL_HOURS

    revenue = energy.mul(price, axis=0).sum()
    total_energy = energy.sum()
    achieved_price = revenue / total_energy
    market_average = revenue.sum() / total_energy.sum()

    return pd.DataFrame({
        "Erzielter Preis [€/MWh]": achieved_price,
        "Erlösfaktor [%]": achieved_price / market_average * 100,
        "Erzeugt bei negativem Preis [%]": energy[price < 0].sum() / total_energy * 100,
    })


# ---------- Datenablage: geladene Daten werden auf dem Rechner gespeichert ----------

CACHE_DIR = Path(__file__).with_name("datenablage")


def _cached_frame(name: str, fetch, start: date, end: date) -> pd.DataFrame:
    """Holt nur die Tage, die noch nicht in der Datenablage stehen, und speichert alles wieder ab.

    Das Enddatum ist nicht enthalten, wie bei den anderen Funktionen.
    """
    path = CACHE_DIR / f"{name}.csv"
    cache = pd.read_csv(path, index_col=0, parse_dates=True) if path.exists() else pd.DataFrame()

    if cache.empty:
        merged = fetch(start, end)
    else:
        merged = cache
        if start < cache.index.min().date():
            merged = pd.concat([fetch(start, cache.index.min().date()), merged])
        if end > cache.index.max().date():
            # Der letzte Tag wird noch einmal geholt, weil er unvollständig gewesen sein kann
            merged = pd.concat([merged, fetch(cache.index.max().date(), end)])

    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    CACHE_DIR.mkdir(exist_ok=True)
    merged.to_csv(path)
    return merged[(merged.index >= pd.Timestamp(start)) & (merged.index < pd.Timestamp(end))]


def cached_generation(start: date, end: date) -> pd.DataFrame:
    """Erzeugung aus der Datenablage, fehlende Tage kommen aus dem Netz."""
    return _cached_frame("erzeugung", fetch_generation, start, end)


def cached_prices(start: date, end: date) -> pd.Series:
    """Strompreise aus der Datenablage, fehlende Tage kommen aus dem Netz."""
    frame = _cached_frame("preise", lambda s, e: fetch_prices(s, e).to_frame(), start, end)
    return frame[PRICE_COLUMN]
