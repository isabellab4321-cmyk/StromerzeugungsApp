"""Streamlit-App: Stromerzeugung auswerten, Erlös berechnen, Prognosen prüfen und von Claude einen Bericht schreiben lassen.

Starten mit:
    streamlit run app.py
"""

import html
import re
from datetime import date, timedelta
from io import BytesIO
from xml.sax.saxutils import escape

import altair as alt
import anthropic
import pandas as pd
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from energy_charts import (
    baseline_last_week,
    compare_forecast,
    compute_price_risk,
    compute_revenue,
    cached_generation,
    cached_prices,
    read_forecast_csv,
)
from kennzahlen_erzeugung import RENEWABLE_PREFIXES, compute_kpis

MODEL = "claude-sonnet-5-5"
AUTOMATIC = "Automatisch aus dem Netz"


def de(value: float, decimals: int = 1, sign: bool = False) -> str:
    """Zahl im deutschen Format: 50.953,0 (Punkt für Tausender, Komma für Nachkommastellen)."""
    text = f"{value:{'+' if sign else ''},.{decimals}f}"
    return text.replace(",", "§").replace(".", ",").replace("§", ".")


def pct(fraction: float, decimals: int = 1) -> str:
    """Anteil als Prozent im deutschen Format, zum Beispiel 0,632 -> 63,2 %."""
    return f"{de(fraction * 100, decimals)} %"


def german_table(frame: pd.DataFrame):
    """Tabelle mit deutschen Zahlen, ohne Hervorhebung."""
    numbers = frame.select_dtypes("number").columns
    return frame.style.format(lambda v: de(v, 1) if isinstance(v, float) and pd.notna(v) else v, subset=numbers)

# Erklärungen für Fachbegriffe: Sie erscheinen, wenn man mit der Maus über das Wort fährt
GLOSSARY = {
    "Erlösfaktor": "Verhältnis des erzielten Preises einer Quelle zum Durchschnitt aller Quellen. "
                   "100 % = Durchschnitt. Über 100 % = bessere Verkaufszeiten, unter 100 % = schlechtere.",
    "Erlös": "Erzeugte Menge mal Börsenpreis. Das ist der reine Markterlös, ohne Förderung und Verträge.",
    "Markterlös": "Erzeugte Menge mal Börsenpreis. Ohne Förderung, Verträge und Kosten.",
    "erzielter Preis": "Durchschnittspreis, zu dem eine Quelle ihren Strom verkauft hat, gewichtet nach der Menge.",
    "Durchschnittspreis": "Mittlerer Preis, zu dem verkauft wurde. Je nach Zusammenhang für eine Quelle oder für den Markt.",
    "Börsenpreis": "Preis, zu dem Strom an der Strombörse gehandelt wird. Hier der Preis für den Tag im Voraus.",
    "negative Preise": "Preise unter null Euro. Wer dann verkauft, zahlt drauf.",
    "negativem Preis": "Preis unter null Euro. Wer dann verkauft, zahlt drauf.",
    "Anteil erneuerbar": "Teil der erneuerbaren Energie an der gesamten Stromerzeugung.",
    "erneuerbar": "Energie aus Quellen, die sich immer wieder erneuern, zum Beispiel Wind, Sonne, Wasser und Biomasse.",
    "Erneuerbare": "Energie aus Quellen, die sich immer wieder erneuern, zum Beispiel Wind, Sonne, Wasser und Biomasse.",
    "Megawatt": "Leistung: wie viel Strom gerade fließt. 1 MW = 1.000 kW.",
    "MW": "Megawatt, eine Leistung: wie viel Strom gerade fließt.",
    "GWh": "Gigawattstunde, eine Energiemenge. 1 GWh = 1.000 MWh.",
    "TWh": "Terawattstunde, eine große Energiemenge. 1 TWh = 1.000 GWh.",
    "Prognose": "Vorhersage, wie viel Strom erzeugt werden wird.",
    "Soll": "Die Prognose: was man erwartet hat.",
    "Ist": "Die tatsächlich gemessene Erzeugung.",
    "Abweichung": "Unterschied zwischen Prognose und tatsächlicher Erzeugung.",
    "Prozentpunkte": "Unterschied zwischen zwei Prozentwerten. Von 50 % auf 55 % sind es 5 Prozentpunkte.",
    "Preisrisiko": "Wie stark Preise schwanken und wie sich das auf die Einnahmen auswirkt.",
    "Day-Ahead": "Handel für den Tag im Voraus.",
    "ESG": "Umwelt, Soziales und Unternehmensführung. Ein Nachhaltigkeitsbericht bewertet diese Bereiche.",
    "Mengeneffekt": "Der Teil der Abweichung, der durch eine andere Strommenge entstanden ist. Gerechnet mit dem Planpreis.",
    "Preiseffekt": "Der Teil der Abweichung, der durch einen anderen Preis entstanden ist. Gerechnet mit der Istmenge.",
    "Plan-Erlös": "Erlös, der nach dem Plan erwartet wurde: geplante Menge mal geplanter Preis.",
    "Deckungsbeitrag": "Erlös minus variable Kosten. Zeigt, wie viel Geld von jedem Euro Erlös übrig bleibt.",
    "Marge": "Deckungsbeitrag in Prozent des Erlöses.",
    "variable Kosten": "Kosten, die mit der erzeugten Strommenge steigen, zum Beispiel Brennstoff.",
}


def highlight_renewable(frame: pd.DataFrame, green_names: set):
    """Erneuerbare Zeilen hell grün hinterlegt, inklusive des Namens des Energieträgers."""
    # Der Name wird zur ersten Spalte, damit auch er grün markiert werden kann
    table = frame.rename_axis("Energieträger").reset_index()
    green_rows = table["Energieträger"].isin(green_names)
    numbers = table.select_dtypes("number").columns
    return (
        table.style.format(lambda v: de(v, 1) if isinstance(v, float) and pd.notna(v) else v, subset=numbers)
        .apply(
            lambda row: ["background-color: rgba(46, 158, 91, 0.14)" if green_rows[row.name] else "" for _ in row],
            axis=1,
        )
        .hide(axis="index")
    )


def build_summary(kpis: dict, revenue) -> str:
    """Fasst die Kennzahlen und den Erlös als Text zusammen, den Claude lesen kann."""
    start, end = kpis["period"]
    # Zeitpunkte als Text, damit die Zahlen-Rundung nur auf Zahlen angewendet wird
    stats = kpis["stats"].copy()
    stats["Zeitpunkt Maximum"] = stats["Zeitpunkt Maximum"].dt.strftime("%d.%m.%Y %H:%M")
    stats = stats.round(1)
    share_min, share_min_time = kpis["renewable_share_min"]
    share_max, share_max_time = kpis["renewable_share_max"]
    total_max, total_max_time = kpis["total_max_mw"]

    return f"""Zeitraum mit vollständigen Daten: {start:%d.%m.%Y %H:%M} bis {end:%d.%m.%Y %H:%M}
Land: Deutschland
Anzahl 15-Minuten-Intervalle: {kpis['n_intervals']}

Anteil erneuerbarer Energie (energiegewichtet): {pct(kpis['renewable_share_energy'], 1)}
Anteil erneuerbarer Energie (Mittel über Intervalle): {pct(kpis['renewable_share_mean'], 1)}
Minimum des Anteils: {pct(share_min, 1)} am {share_min_time:%d.%m.%Y %H:%M}
Maximum des Anteils: {pct(share_max, 1)} am {share_max_time:%d.%m.%Y %H:%M}

Gesamterzeugung Mittelwert: {de(kpis['total_mean_mw'], 1)} MW
Gesamterzeugung Maximum: {de(total_max, 1)} MW am {total_max_time:%d.%m.%Y %H:%M}
Gesamterzeugung Energie: {de(kpis['total_energy_twh'], 2)} TWh

Pro Energieträger (Mittelwert in MW, Maximum in MW mit Zeitpunkt, Energie in GWh):
{stats.to_string()}

Markterlös (Menge mal Börsenpreis) pro Energieträger:
{revenue.round(1).to_string()}
"""


def ask_claude(api_key: str, summary: str) -> str:
    """Schickt die Zusammenfassung an die Claude API und gibt den Bericht zurück."""
    client = anthropic.Anthropic(api_key=api_key)
    prompt = (
        "Du bist Nachhaltigkeitsberater und schreibst einen ESG-Bericht auf Deutsch "
        "(Umwelt-Teil, Schwerpunkt Stromerzeugung). Gliedere den Bericht in diese Abschnitte: "
        "Zusammenfassung, Kennzahlen zur Energiewende, Erlös und Preise, Trends, Auffälligkeiten, "
        "Ausblick und Fazit. Formulierung sachlich und förmlich, etwa 300 Wörter. "
        "Nutze nur die Zahlen unten und erfinde keine Werte. Erkläre Fachbegriffe wie ESG kurz, "
        "denn die Leserin ist Anfängerin. Weise am Ende darauf hin, dass sich der Bericht nur auf "
        "die Daten im Datensatz bezieht und keine vollständige Nachhaltigkeitsbewertung ist.\n\n"
        f"Kennzahlen:\n{summary}"
    )
    response = client.messages.create(
        model=MODEL,
        # Großzügiges Limit: Denkschritte zählen mit, und ein zu kleines Limit schneidet den Text ab
        max_tokens=6000,
        messages=[{"role": "user", "content": prompt}],
    )
    # Die Antwort besteht aus mehreren Teilen. Manche sind interne Denkschritte ohne Text,
    # deshalb sammeln wir nur die Teile, die echten Text enthalten.
    text = "".join(block.text for block in response.content if block.type == "text")
    if response.stop_reason == "max_tokens":
        text += "\n\n(Hinweis: Die Antwort wurde abgeschnitten. Bitte erneut erstellen.)"
    return text


def report_to_pdf(report: str) -> bytes:
    """Schreibt den Berichtstext in eine PDF-Datei und gibt deren Inhalt zurück."""
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Bericht: Stromerzeugung in Deutschland", styles["Title"]),
        Spacer(1, 12),
    ]
    for line in report.splitlines():
        text = line.strip()
        if not text:
            story.append(Spacer(1, 6))
        elif text.startswith("#"):
            # Überschriften aus dem Markdown-Text (z. B. "## Trends") werden zu fetten Titeln
            story.append(Paragraph(escape(text.lstrip("#").strip()), styles["Heading2"]))
        elif text.startswith(("- ", "* ")):
            story.append(Paragraph("• " + escape(text[2:].replace("**", "")), styles["BodyText"]))
        else:
            story.append(Paragraph(escape(text.replace("**", "")), styles["BodyText"]))

    buffer = BytesIO()
    SimpleDocTemplate(buffer, pagesize=A4, leftMargin=50, rightMargin=50, topMargin=50, bottomMargin=50).build(story)
    return buffer.getvalue()


@st.cache_data(ttl=3600)  # speichert das Ergebnis eine Stunde, damit nicht bei jedem Klick neu geladen wird
def load_generation_online(start: date, end: date):
    # Die Datenablage holt nur die fehlenden Tage aus dem Netz
    return cached_generation(start, end)


@st.cache_data(ttl=3600)
def load_prices_online(start: date, end: date):
    return cached_prices(start, end)


def hover_text(text: str) -> str:
    """Setzt in den Text für jeden Fachbegriff eine Markierung, die beim Darüberfahren die Erklärung zeigt."""
    # Längere Begriffe zuerst, damit z. B. "Erlösfaktor" nicht als "Erlös" erkannt wird
    terms = sorted(GLOSSARY, key=len, reverse=True)
    pattern = re.compile(r"(?<!\w)(" + "|".join(re.escape(term) for term in terms) + r")(?!\w)")

    def wrap(match: re.Match) -> str:
        word = match.group(1)
        tip = html.escape(GLOSSARY[word])
        # tabindex="0" macht das Wort anklickbar; die Erklärung erscheint beim Klick
        return f'<span class="term" tabindex="0">{word}<span class="tip">{tip}</span></span>'

    return pattern.sub(wrap, text)


def explain(text: str) -> None:
    """Erklärung in einem aufklappbaren Bereich, Fachbegriffe sind anklickbar."""
    with st.expander("Methode und Hinweise"):
        st.markdown(hover_text(text), unsafe_allow_html=True)


def csv_download(frame: pd.DataFrame, file_name: str) -> None:
    """Knopf zum Herunterladen einer Tabelle als CSV, mit Semikolon und Komma, wie in deutschem Excel."""
    st.download_button(
        "Tabelle als CSV herunterladen",
        data=frame.to_csv(sep=";", decimal=",").encode("utf-8-sig"),
        file_name=file_name,
        mime="text/csv",
    )


def summary_numbers(kpis: dict, revenue: pd.DataFrame) -> dict:
    """Die wichtigsten Zahlen eines Zeitraums, damit man zwei Zeiträume vergleichen kann."""
    total_revenue = revenue["Erlös [Mio. €]"].sum()
    total_energy_gwh = revenue["Energie [GWh]"].sum()
    return {
        "share": kpis["renewable_share_energy"],
        "twh": kpis["total_energy_twh"],
        "revenue": total_revenue,
        "price": total_revenue * 1000 / total_energy_gwh,
    }


def set_period(start: date, end: date) -> None:
    """Setzt den Zeitraum über die Schnellauswahl (wird vor dem Neuaufbau der Seite ausgeführt)."""
    st.session_state["von"] = start
    st.session_state["bis"] = end


def show_year_comparison(
    current: dict, previous: dict | None, prev_start: date, prev_end: date, error: str = ""
) -> None:
    """Vergleich des gewählten Zeitraums mit demselben Zeitraum ein Jahr früher."""
    st.subheader("Vergleich mit dem Vorjahr")
    if previous is None:
        st.info(f"Für den Vorjahreszeitraum sind keine Daten verfügbar. {error}".strip())
        return

    rows = {
        "Anteil erneuerbar [%]": (current["share"] * 100, previous["share"] * 100, "Prozentpunkte"),
        "Erzeugung [TWh]": (current["twh"], previous["twh"], "%"),
        "Erlös [Mio. €]": (current["revenue"], previous["revenue"], "%"),
        "Ø erzielter Preis [€/MWh]": (current["price"], previous["price"], "%"),
    }
    table = pd.DataFrame(
        {
            "Aktuell": {name: values[0] for name, values in rows.items()},
            "Vorjahr": {name: values[1] for name, values in rows.items()},
            "Veränderung": {
                name: f"{de(values[0] - values[1], 1, sign=True)} Prozentpunkte" if values[2] == "Prozentpunkte"
                else f"{de((values[0] / values[1] - 1) * 100, 1, sign=True)} %"
                for name, values in rows.items()
            },
        }
    )
    st.dataframe(german_table(table.round(1)))
    explain(
        f"Vorjahreszeitraum: {prev_start:%d.%m.%Y} bis {prev_end - timedelta(days=1):%d.%m.%Y}. "
        "Die Veränderung zeigt, wie sich die Zahl gegenüber dem Vorjahr entwickelt hat."
    )


def show_start(kpis: dict) -> None:
    """Reiter Erzeugung: Anteil erneuerbarer Energie über die Zeit und die Tabelle pro Energieträger."""
    st.markdown(f"**Erneuerbare Energien lieferten {pct(kpis['renewable_share_energy'], 1)} des Stroms im Zeitraum.**")
    daily_share = kpis["share_daily"].dropna()
    c1, c2, c3 = st.columns(3)
    c1.metric("Ø Tagesanteil erneuerbar", pct(daily_share.mean(), 1))
    c2.metric("Höchster Tagesanteil", pct(daily_share.max(), 1))
    c3.metric("Niedrigster Tagesanteil", pct(daily_share.min(), 1))

    st.subheader("Anteil erneuerbarer Energie (Tagesmittel, in %)")
    daily = (kpis["share_daily"].dropna() * 100).rename("Anteil erneuerbar [%]")
    st.line_chart(daily)
    explain(
        "Jeder Punkt ist der Anteil erneuerbarer Energie an der gesamten Stromerzeugung eines Tages. "
        "Erneuerbar sind Wind, Sonne, Wasserkraft, Biomasse und einige kleinere Quellen. "
        "Hohe Werte bedeuten: Ein großer Teil des Stroms kam aus erneuerbaren Quellen."
    )

    st.subheader("Erzeugung pro Energieträger")
    table = kpis["stats"].drop(columns="Zeitpunkt Maximum").round(1)
    renewable = table.index.str.startswith(RENEWABLE_PREFIXES)
    # Erneuerbare zuerst, jeweils nach Energiemenge sortiert
    green = table[renewable].sort_values("Energie [GWh]", ascending=False)
    other = table[~renewable].sort_values("Energie [GWh]", ascending=False)

    # Summen: Mittelwerte und Energie lassen sich addieren, Höchstwerte nicht, deshalb bleiben sie leer
    def total_row(name: str, frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Mittelwert [MW]": [frame["Mittelwert [MW]"].sum()],
                "Maximum [MW]": [float("nan")],
                "Energie [GWh]": [frame["Energie [GWh]"].sum()],
            },
            index=[name],
        )

    totals = pd.concat([
        total_row("Summe erneuerbar", green),
        total_row("Summe nicht erneuerbar", other),
        total_row("Gesamt", table),
    ]).round(1)

    # Erneuerbare Zeilen werden komplett grün hinterlegt
    shown = pd.concat([green, other, totals])
    st.dataframe(highlight_renewable(shown, set(green.index)))
    csv_download(shown, "erzeugung_pro_energietraeger.csv")
    explain(
        "Grün hinterlegte Zeilen sind erneuerbare Energie. Mittelwert und Maximum sind Leistungen in "
        "Megawatt (MW). Energie in GWh ist die insgesamt erzeugte Menge im Zeitraum."
    )

    st.subheader("Erzeugte Energie pro Energieträger (GWh)")
    renewable_bar_chart(kpis["stats"]["Energie [GWh]"], "Energie [GWh]")
    explain(
        "Jeder Balken zeigt, wie viel Strom eine Quelle im Zeitraum erzeugt hat. "
        "Grün sind die erneuerbaren Energieträger, grau die anderen."
    )


def show_key_figures(kpis: dict, current: dict, previous: dict | None) -> None:
    """Reiter 2: Kennzahlen zur Erzeugung."""
    st.subheader("Kennzahlen")
    # Veränderung gegenüber dem Vorjahr, falls Vorjahresdaten vorhanden sind
    share_delta = energy_delta = None
    if previous is not None:
        share_delta = f"{de((current['share'] - previous['share']) * 100, 1, sign=True)} Prozentpunkte"
        energy_delta = f"{de((current['twh'] / previous['twh'] - 1) * 100, 1, sign=True)} % zum Vorjahr"
    col1, col2, col3 = st.columns(3)
    col1.metric("Anteil erneuerbar (energiegewichtet)", f"{pct(kpis['renewable_share_energy'], 1)}", delta=share_delta)
    col2.metric("Gesamterzeugung Mittel", f"{de(kpis['total_mean_mw'], 1)} MW")
    col3.metric("Gesamterzeugung Energie", f"{de(kpis['total_energy_twh'], 1)} TWh", delta=energy_delta)



def renewable_bar_chart(values: pd.Series, axis_title: str) -> None:
    """Senkrechte Balken: erneuerbare Energieträger grün, die anderen grau."""
    data = pd.DataFrame({
        "Energieträger": values.index,
        "Wert": values.values,
        "Erneuerbar": ["Ja" if name.startswith(RENEWABLE_PREFIXES) else "Nein" for name in values.index],
    })
    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            # Senkrechte Balken, die Namen werden schräg gestellt, damit sie lesbar bleiben
            x=alt.X("Energieträger:N", sort="-y", title=None, axis=alt.Axis(labelAngle=-45, labelOverlap=False)),
            y=alt.Y("Wert:Q", title=axis_title),
            color=alt.Color(
                "Erneuerbar:N",
                scale=alt.Scale(domain=["Ja", "Nein"], range=["#2e9e5b", "#9aa0a6"]),
                legend=alt.Legend(title="Erneuerbar"),
            ),
            tooltip=["Energieträger", alt.Tooltip("Wert:Q", format=",.1f")],
        )
        .properties(height=420)
    )
    st.altair_chart(chart, width="stretch")


def monthly_table(generation: pd.DataFrame, prices: pd.Series) -> pd.DataFrame:
    """Erlös, Energie und Anteil erneuerbar pro Monat, passend für Finanzberichte."""
    sources = generation.drop(columns=["Energy storage"], errors="ignore").dropna(axis=1, how="all")
    joined = sources.join(prices, how="inner").dropna()
    price = joined[prices.name]
    energy = joined.drop(columns=prices.name) * 0.25  # jede Viertelstunde dauert 0,25 Stunden
    renewable = [c for c in energy.columns if c.startswith(RENEWABLE_PREFIXES)]
    total_mwh = energy.sum(axis=1)
    month = joined.index.to_period("M")
    table = pd.DataFrame({
        "Energie [GWh]": total_mwh.groupby(month).sum() / 1000,
        "Erlös [Mio. €]": (total_mwh * price).groupby(month).sum() / 1_000_000,
        "Anteil erneuerbar [%]": energy[renewable].sum(axis=1).groupby(month).sum()
        / total_mwh.groupby(month).sum() * 100,
    })
    table["Ø erzielter Preis [€/MWh]"] = table["Erlös [Mio. €]"] * 1000 / table["Energie [GWh]"]
    table.index = table.index.strftime("%m.%Y")
    table.index.name = "Monat"
    return table.round(1)


def show_revenue(revenue, monthly: pd.DataFrame) -> None:
    """Reiter 3: Erlöse aus Erzeugung mal Börsenpreis."""
    total_revenue = revenue["Erlös [Mio. €]"].sum()
    total_energy_gwh = revenue["Energie [GWh]"].sum()
    average_price = total_revenue * 1000 / total_energy_gwh
    st.markdown(
        f"**Der Erlös betrug insgesamt {de(total_revenue, 0)} Mio. €, im Schnitt {de(average_price, 1)} €/MWh.**"
    )

    st.subheader("Erlös")
    col1, col2, col3 = st.columns(3)
    col1.metric("Erlös gesamt", f"{de(total_revenue, 1)} Mio. €")
    col2.metric("Ø erzielter Preis", f"{de(average_price, 1)} €/MWh")
    col3.metric("Energie gesamt", f"{de(total_energy_gwh, 1)} GWh")

    st.subheader("Erlös pro Energieträger (Mio. €)")
    renewable_bar_chart(revenue["Erlös [Mio. €]"], "Erlös [Mio. €]")
    explain(
        "Jeder Balken zeigt, wie viel Geld eine Energiequelle im Zeitraum eingebracht hat. "
        "Je länger der Balken, desto mehr Erlös. Grün sind die erneuerbaren Energieträger, grau die anderen."
    )

    st.subheader("Tabelle pro Energieträger")
    # Zeile unten: alle Energieträger zusammen, mit dem Durchschnittspreis über alle Quellen
    overall = pd.DataFrame(
        {
            "Energie [GWh]": [total_energy_gwh],
            "Erlös [Mio. €]": [total_revenue],
            "Erzielter Preis [€/MWh]": [average_price],
        },
        index=["Alle Energieträger"],
    )
    table = pd.concat([revenue, overall]).round(1)
    table = table.rename(columns={"Erzielter Preis [€/MWh]": "Durchschnittspreis je Energieträger [€/MWh]"})
    # Erneuerbare zuerst, wie auf der Titelseite, dann die anderen, zum Schluss die Zeile "Alle"
    renewable = table.index.str.startswith(RENEWABLE_PREFIXES)
    green = table[renewable].sort_values("Erlös [Mio. €]", ascending=False)
    other = table[~renewable & (table.index != "Alle Energieträger")].sort_values("Erlös [Mio. €]", ascending=False)
    shown = pd.concat([green, other, table.loc[["Alle Energieträger"]]])
    st.dataframe(highlight_renewable(shown, set(green.index)))
    csv_download(shown, "erloese_pro_energietraeger.csv")
    explain(
        "Energie in GWh ist die erzeugte Menge. Erlös in Mio. € ist die Menge mal Börsenpreis. "
        "Der Durchschnittspreis je Energieträger zeigt, zu welchem Preis die Quelle im Schnitt verkauft hat. "
        "Die Zeile „Alle Energieträger“ zeigt den Durchschnitt über alle Quellen zusammen."
    )
    explain(
        "Der Erlös ist der reine Markterlös: Menge mal Börsenpreis. Förderungen, Verträge und Kosten "
        "sind nicht enthalten."
    )

    st.subheader("Deckungsbeitrag")
    st.write(
        "Hier trägst du die **variablen Kosten** je Energieträger ein. Sie sind am Anfang 0, "
        "weil die Kosten bekannt sein müssen. Der Deckungsbeitrag ist der Erlös minus diese Kosten."
    )
    cost_input = pd.DataFrame({
        "Energieträger": revenue.index,
        "Variable Kosten [€/MWh]": [0.0] * len(revenue.index),
    })
    costs = st.data_editor(
        cost_input, hide_index=True, disabled=["Energieträger"], key="cost_editor", width="stretch"
    ).set_index("Energieträger")["Variable Kosten [€/MWh]"].astype(float)

    margin = pd.DataFrame({
        "Erlös [Mio. €]": revenue["Erlös [Mio. €]"],
        "Kosten [Mio. €]": revenue["Energie [GWh]"] * costs / 1000,
    })
    margin["Deckungsbeitrag [Mio. €]"] = margin["Erlös [Mio. €]"] - margin["Kosten [Mio. €]"]
    margin["Marge [%]"] = (margin["Deckungsbeitrag [Mio. €]"] / margin["Erlös [Mio. €]"] * 100).where(
        margin["Erlös [Mio. €]"] > 0
    )
    total = margin.sum()
    total["Marge [%]"] = total["Deckungsbeitrag [Mio. €]"] / total["Erlös [Mio. €]"] * 100
    margin.loc["Alle Energieträger"] = total
    renewable = margin.index.str.startswith(RENEWABLE_PREFIXES)
    green = margin[renewable].sort_values("Deckungsbeitrag [Mio. €]", ascending=False)
    other = margin[~renewable & (margin.index != "Alle Energieträger")].sort_values(
        "Deckungsbeitrag [Mio. €]", ascending=False
    )
    shown = pd.concat([green, other, margin.loc[["Alle Energieträger"]]]).round(1)
    st.markdown(
        f"**Der Deckungsbeitrag beträgt insgesamt {de(total['Deckungsbeitrag [Mio. €]'], 1)} Mio. €, "
        f"das ist eine Marge von {de(total['Marge [%]'], 1)} %.**"
    )
    st.dataframe(highlight_renewable(shown, set(green.index)))
    csv_download(shown, "deckungsbeitrag.csv")
    explain(
        "Der **Deckungsbeitrag** ist der Erlös minus die **variablen Kosten**. Die **Marge** ist der "
        "Deckungsbeitrag in Prozent des Erlöses. Feste Kosten wie Miete oder Personal sind nicht enthalten."
    )




def show_price_risk_by_source(prices: pd.Series, df) -> None:
    """Preisrisiko pro Energieträger: erzielter Preis, Erlösfaktor und Anteil bei negativen Preisen."""
    st.subheader("Preisrisiko pro Energieträger")
    risk = compute_price_risk(df, prices).round(1)
    renewable = risk.index.str.startswith(RENEWABLE_PREFIXES)
    green = risk[renewable].sort_values("Erlösfaktor [%]", ascending=False)
    other = risk[~renewable].sort_values("Erlösfaktor [%]", ascending=False)
    shown = pd.concat([green, other])
    st.dataframe(highlight_renewable(shown, set(green.index)))
    csv_download(shown, "preisrisiko_pro_energietraeger.csv")
    explain(
        "**Erlösfaktor:** 100 % bedeutet, dass die Quelle genau den Durchschnittspreis aller Quellen erzielt. "
        "Über 100 % heißt: Sie verkauft zu besseren Zeiten. Unter 100 % heißt: Sie verkauft zu schlechteren Zeiten. "
        "**Erzeugt bei negativem Preis:** Anteil der Erzeugung, der zu Zeiten mit negativem Preis entstand. "
        "Grün hinterlegt sind die erneuerbaren Energieträger."
    )

    renewable_bar_chart(risk["Erlösfaktor [%]"], "Erlösfaktor [%]")
    explain("Die Balkenhöhe zeigt den Erlösfaktor. 100 % entspricht dem Durchschnitt aller Quellen.")


def show_source_prices(prices: pd.Series, df, technology: str) -> None:
    """Preise für einen ausgewählten Energieträger: Menge, erzielter Preis und Verlauf."""
    st.subheader(f"Preise für {technology}")
    joined = df[[technology]].join(prices, how="inner").dropna()
    if joined.empty:
        st.warning("Für diesen Energieträger gibt es im Zeitraum keine Werte mit Preis.")
        return

    energy_mwh = joined[technology] * 0.25  # jede Viertelstunde dauert 0,25 Stunden
    price = joined[prices.name]
    total_energy = energy_mwh.sum()
    achieved_price = (energy_mwh * price).sum() / total_energy

    risk = compute_price_risk(df, prices)
    if technology not in risk.index:
        st.warning("Für diesen Energieträger lässt sich das Preisrisiko nicht berechnen.")
        return
    factor = risk.loc[technology, "Erlösfaktor [%]"]
    st.markdown(
        f"**{technology} erzielte {de(achieved_price, 1)} €/MWh. Das sind {de(factor, 1)} % "
        "des Durchschnitts aller Quellen.**"
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Erzielter Preis", f"{de(achieved_price, 1)} €/MWh")
    col2.metric("Erlösfaktor", f"{de(risk.loc[technology, 'Erlösfaktor [%]'], 1)} %")
    col3.metric("Erzeugt bei negativem Preis", f"{de(risk.loc[technology, 'Erzeugt bei negativem Preis [%]'], 1)} %")
    explain(
        "Der **Erlösfaktor** zeigt, wie gut die Quelle verkauft: 100 % = Durchschnitt aller Quellen. "
        "Der **erzielte Preis** ist der Durchschnittspreis, zu dem die Quelle verkauft hat."
    )

    # Tageswerte: erzielter Preis dieser Quelle gegen den Durchschnitt des Marktes
    day = joined.index.normalize()
    achieved_daily = (energy_mwh * price).groupby(day).sum() / energy_mwh.groupby(day).sum()
    market_daily = prices.resample("D").mean()
    st.line_chart(pd.DataFrame({
        "Erzielter Preis": achieved_daily,
        "Ø Börsenpreis": market_daily,
    }).dropna())
    explain(
        f"Die Linien zeigen pro Tag, zu welchem Preis {technology} verkauft hat (erzielter Preis) und den "
        "Durchschnitt des Marktes. Liegt die Linie des erzielten Preises unter dem Markt, erzeugte die Quelle "
        "überwiegend zu ungünstigen Zeiten."
    )


def show_prices(prices: pd.Series, df) -> None:
    """Reiter: Börsenpreise über die Zeit, Monatsdurchschnitte und Risiko durch negative Preise."""
    technologies = [c for c in df.columns if c != "Energy storage" and df[c].notna().any()]
    technology = st.selectbox("Energieträger", technologies, key="price_source")
    show_source_prices(prices, df, technology)

    show_price_risk_by_source(prices, df)
    st.subheader("Börsenpreise im Zeitraum")
    negative_hours = (prices < 0).sum() * 0.25  # jede Viertelstunde dauert 0,25 Stunden
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ø Börsenpreis", f"{de(prices.mean(), 1)} €/MWh")
    col2.metric("Höchstpreis", f"{de(prices.max(), 1)} €/MWh")
    col3.metric("Tiefstpreis", f"{de(prices.min(), 1)} €/MWh")
    col4.metric("Stunden mit negativem Preis", f"{de(negative_hours, 1)} h")

    st.subheader("Börsenpreis über die Zeit (Tagesmittel, €/MWh)")
    st.line_chart(prices.resample("D").mean().rename("Ø Preis [€/MWh]"))
    explain(
        "Jeder Punkt ist der Durchschnittspreis eines Tages. Starke Ausschläge zeigen, dass der Preis "
        "stark schwankt. Das ist ein Risiko für die Erlöse."
    )


def plan_variance(ist: pd.DataFrame, previous: pd.DataFrame | None) -> None:
    """Reiter Plan gegen Ist: Abweichung des Erlöses, zerlegt in Mengen- und Preiseffekt."""
    st.subheader("Plan gegen Ist")
    st.write(
        "Der **Plan** ist standardmäßig der Vorjahreswert pro Monat. Er lässt sich in der Tabelle "
        "überschreiben, zum Beispiel mit eigenen Annahmen. Die Abweichung zeigt, was sich gegenüber dem Plan "
        "geändert hat."
    )

    # Plan pro Monat: Vorjahreswert, falls vorhanden, sonst der Istwert als Ausgangspunkt
    rows = []
    for month in ist.index:
        last_year = f"{month[:3]}{int(month[3:]) - 1}"
        if previous is not None and last_year in previous.index:
            base = previous.loc[last_year]
        else:
            base = ist.loc[month]
        rows.append({
            "Monat": month,
            "Plan Energie [GWh]": round(float(base["Energie [GWh]"]), 1),
            "Plan Preis [€/MWh]": round(float(base["Ø erzielter Preis [€/MWh]"]), 1),
        })
    plan = st.data_editor(
        pd.DataFrame(rows),
        hide_index=True,
        disabled=["Monat"],
        key="plan_editor",
        width="stretch",
        column_config={
            # "localized" zeigt Zahlen in der Schreibweise des Browsers an, bei Deutsch also 9.800
            "Plan Energie [GWh]": st.column_config.NumberColumn(format="localized"),
            "Plan Preis [€/MWh]": st.column_config.NumberColumn(format="localized"),
        },
    ).set_index("Monat").astype(float)

    plan_e = plan["Plan Energie [GWh]"]
    plan_p = plan["Plan Preis [€/MWh]"]
    ist_e = ist["Energie [GWh]"]
    ist_rev = ist["Erlös [Mio. €]"]
    # Istpreis aus Erlös und Menge berechnen, damit die Zerlegung genau aufgeht (nicht gerundet übernehmen)
    ist_p = ist_rev * 1000 / ist_e
    plan_rev = plan_e * plan_p / 1000  # Mio. €
    volume = (ist_e - plan_e) * plan_p / 1000
    price = (ist_p - plan_p) * ist_e / 1000
    diff = ist_rev - plan_rev

    result = pd.DataFrame({
        "Plan-Erlös [Mio. €]": plan_rev,
        "Ist-Erlös [Mio. €]": ist_rev,
        "Mengeneffekt [Mio. €]": volume,
        "Preiseffekt [Mio. €]": price,
        "Abweichung [Mio. €]": diff,
        "Abweichung [%]": diff / plan_rev * 100,
    })
    total = result.drop(columns="Abweichung [%]").sum()
    total["Abweichung [%]"] = total["Abweichung [Mio. €]"] / total["Plan-Erlös [Mio. €]"] * 100
    result.loc["Gesamt"] = total
    result = result.round(1)

    gesamt = result.loc["Gesamt"]
    treiber = "Mengeneffekt" if abs(gesamt["Mengeneffekt [Mio. €]"]) >= abs(gesamt["Preiseffekt [Mio. €]"]) else "Preiseffekt"
    st.markdown(
        f"**Der Erlös weicht um {de(gesamt['Abweichung [Mio. €]'], 1)} Mio. € "
        f"({de(gesamt['Abweichung [%]'], 1)} %) vom Plan ab. Der größere Einfluss ist der {treiber}.**"
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Plan-Erlös gesamt", f"{de(gesamt['Plan-Erlös [Mio. €]'], 1)} Mio. €")
    c2.metric("Ist-Erlös gesamt", f"{de(gesamt['Ist-Erlös [Mio. €]'], 1)} Mio. €")
    c3.metric("Abweichung", f"{de(gesamt['Abweichung [Mio. €]'], 1)} Mio. €")
    st.bar_chart(result.drop(index="Gesamt")[["Mengeneffekt [Mio. €]", "Preiseffekt [Mio. €]"]])
    st.dataframe(german_table(result))
    csv_download(result, "plan_gegen_ist.csv")
    explain(
        "Der **Mengeneffekt** zeigt, wie viel die andere Strommenge bewirkt hat, gerechnet mit dem Planpreis. "
        "Der **Preiseffekt** zeigt, wie viel der andere Preis bewirkt hat, gerechnet mit der Istmenge. "
        "Zusammen ergeben beide die Abweichung. Der **Plan-Erlös** ist geplante Menge mal geplanter Preis."
    )


def show_forecast(df) -> None:
    """Reiter 4: Prognose (Soll) gegen echte Werte (Ist)."""
    st.subheader("Soll gegen Ist")
    st.write(
        "Hier wird gezeigt, wie gut eine Prognose zur echten Erzeugung passt. "
        "**Ist** sind die gemessenen Werte, **Soll** ist die Prognose."
    )

    # Alle Energieträger mit Messwerten, Solar steht vorne, weil es als Beispiel am aussagekräftigsten ist
    technologies = [c for c in df.columns if c != "Energy storage" and df[c].notna().any()]
    technologies.sort(key=lambda name: name != "Solar")
    if not technologies:
        st.warning("Für diesen Datensatz gibt es keine Messwerte.")
        return
    technology = st.selectbox("Energieträger", technologies)
    option = st.radio(
        "Prognose aus",
        ["Vergleich mit der Vorwoche (einfacher Vergleichswert)", "Eigene Prognose‑Datei (CSV)"],
    )
    explain(
        "**Wer macht die Prognose?** Bei der Vorwoche berechnet die App den Vergleichswert selbst. Das ist "
        "keine professionelle Prognose eines Anbieters. Bei der eigenen Datei stammt die Prognose von der "
        "Quelle, die hochgeladen wird.\n\n"
        "**So wird die Vorwochen‑Prognose berechnet:** Die Prognose für eine Viertelstunde ist der echte "
        "Messwert von genau sieben Tagen davor, zur gleichen Uhrzeit und am gleichen Wochentag. "
        "Ein Beispiel: Die Prognose für Donnerstag, 12:00 Uhr, ist der gemessene Wert vom Donnerstag "
        "der Vorwoche, 12:00 Uhr."
    )

    if option.startswith("Eigene"):
        explain(
            "Die CSV‑Datei braucht zwei Spalten: erste Spalte Zeit (z. B. 2026‑10‑18 12:00), "
            "zweite Spalte Prognose in Megawatt (MW)."
        )
        file = st.file_uploader("Prognose‑Datei hochladen (.csv)", type=["csv"])
        if file is None:
            st.info("Bitte eine CSV‑Datei mit der Prognose hochladen.")
            return
        try:
            forecast = read_forecast_csv(file)
        except Exception as error:
            st.error(f"Die Prognose‑Datei konnte nicht gelesen werden: {error}")
            return
    else:
        forecast = baseline_last_week(df[technology])

    joined, metrics = compare_forecast(df[technology], forecast)
    if metrics["anzahl"] == 0:
        st.warning("Prognose und echte Werte haben keine gemeinsamen Zeitpunkte.")
        return

    st.markdown(
        f"**Die Prognose für {technology} wich im Schnitt um {de(metrics['abweichung_prozent'], 1)} % "
        "von der tatsächlichen Erzeugung ab.**"
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Abweichung (in % der Menge)", f"{de(metrics['abweichung_prozent'], 1)} %")
    col2.metric("Mittlere Abweichung pro Viertelstunde", f"{de(metrics['mittlere_abweichung_mw'], 1)} MW")
    col3.metric("Richtung", "Prognose zu hoch" if metrics["richtung_mw"] > 0 else "Prognose zu niedrig")

    st.line_chart(joined)
    explain(
        "Die Linien zeigen die echten Werte (Ist) und die Prognose (Soll) in Megawatt. Liegen sie eng "
        "beieinander, war die Prognose gut. Die Abweichung in Prozent zeigt den Fehler im Verhältnis zur "
        "gesamten Menge. Die Richtung zeigt, ob die Prognose im Schnitt zu hoch oder zu niedrig war."
    )


def show_report(kpis: dict, revenue, source: str, api_key: str) -> None:
    """Reiter 5: Bericht von Claude erstellen und als PDF herunterladen."""
    st.subheader("Bericht von Claude")
    st.markdown("**Mit einem Claude‑Schlüssel lässt sich hier ein Bericht im ESG‑Stil erstellen.**")
    if not api_key:
        st.warning("Bitte den API‑Schlüssel links in der Seitenleiste eintragen.")
        return
    report_key = f"{source}-{kpis['period'][0]}-{kpis['period'][1]}"
    if st.button("Bericht erstellen"):
        with st.spinner("Claude schreibt den Bericht ..."):
            try:
                # Der Bericht wird in der Sitzung gespeichert, damit er beim Herunterladen nicht verschwindet
                st.session_state["report"] = {
                    "key": report_key,
                    "text": ask_claude(api_key, build_summary(kpis, revenue)),
                }
            except anthropic.APIError as error:
                st.error(f"Die Claude API hat einen Fehler gemeldet: {error}")

    saved = st.session_state.get("report")
    if saved and saved["key"] == report_key:
        st.markdown(saved["text"])
        st.download_button(
            "Bericht als PDF herunterladen",
            data=report_to_pdf(saved["text"]),
            file_name="bericht_stromerzeugung.pdf",
            mime="application/pdf",
        )


def main() -> None:
    # page_icon ist das Symbol im Browser-Tab, hier ein Blitz für Strom
    st.set_page_config(page_title="Stromerzeugung Kennzahlen", page_icon="⚡", layout="wide")
    # Gestaltung: Kennzahlen als Kästen mit grünem Rand, die Farbe funktioniert im hellen und dunklen Design
    st.markdown(
        """
        <style>
        .term {
            border-bottom: 1px dotted #7fd39b;
            cursor: help;
            position: relative;
        }
        .term .tip {
            visibility: hidden;
            opacity: 0;
            position: absolute;
            bottom: 130%;
            left: 0;
            width: 280px;
            background: #1f2d25;
            color: #e8f5ee;
            padding: 8px 10px;
            border-radius: 8px;
            font-size: 0.85rem;
            line-height: 1.35;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
            z-index: 100;
            transition: opacity 0.15s;
        }
        .term:hover .tip,
        .term:focus .tip {
            visibility: visible;
            opacity: 1;
        }
        .term:focus {
            outline: none;
        }
        [data-testid="stMetric"] {
            background: rgba(46, 158, 91, 0.08);
            border-left: 4px solid #2e9e5b;
            border-radius: 12px;
            padding: 12px 16px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Stromerzeugung in Deutschland")
    st.markdown(
        "Diese App wertet Stromerzeugung und Strompreise in Deutschland aus. "
        "Für den Bericht wird ein eigener **Claude API‑Schlüssel** benötigt, der links eingetragen wird."
    )
    # Hier steht die Quellenangabe direkt unter dem Titel; sie wird unten gefüllt, sobald die Quelle feststeht

    # Einstellungen in der Seitenleiste
    # Ausklappbarer Bereich: Datenquelle, Datum und API-Schlüssel, zuklappbar
    settings = st.sidebar.expander("Einstellungen: Datum und API-Schlüssel", expanded=True)
    source = AUTOMATIC
    api_key = settings.text_input("Claude API-Schlüssel", type="password")
    settings.caption("Der Schlüssel wird nur für diese Sitzung benutzt und nirgends gespeichert.")

    today = date.today()
    # Standardmäßig ein ganzes Jahr, damit man die Jahresentwicklung sieht
    st.session_state.setdefault("von", date(2025, 10, 19))
    st.session_state.setdefault("bis", date(2026, 10, 19))
    settings.markdown("**Schnellauswahl**")
    quick = settings.columns(3)
    quick[0].button("Monat", on_click=set_period, args=(today - timedelta(days=30), today))
    quick[1].button("Quartal", on_click=set_period, args=(today - timedelta(days=91), today))
    quick[2].button("Jahr", on_click=set_period, args=(today - timedelta(days=365), today))
    start = settings.date_input("Von", key="von", max_value=today)
    end = settings.date_input("Bis", key="bis", max_value=today)
    if start >= end:
        st.error("Das Startdatum muss vor dem Enddatum liegen.")
        return
    try:
        # Das Enddatum zählt mit, deshalb geben wir einen Tag dazu
        df = load_generation_online(start, end + timedelta(days=1))
    except Exception as error:  # zeigt die Fehlermeldung in der App statt eines Absturzes
        st.error(f"Die Daten konnten nicht aus dem Netz geladen werden: {error}")
        return
    if df.empty:
        st.warning("Für diesen Zeitraum gibt es keine Daten.")
        return

    try:
        kpis = compute_kpis(df)
    except Exception as error:
        st.error(f"Die Kennzahlen konnten nicht berechnet werden: {error}")
        return

    # Börsenpreise für denselben Zeitraum holen und den Erlös berechnen
    period_start = kpis["period"][0].date()
    period_end = kpis["period"][1].date() + timedelta(days=1)
    try:
        prices = load_prices_online(period_start, period_end)
        revenue = compute_revenue(df, prices)
    except Exception as error:
        st.error(f"Die Strompreise konnten nicht geladen werden: {error}")
        return

    # Vorjahreszeitraum: gleiche Tage, ein Jahr früher
    prev_start = (pd.Timestamp(period_start) - pd.DateOffset(years=1)).date()
    prev_end = (pd.Timestamp(period_end) - pd.DateOffset(years=1)).date()
    try:
        prev_gen = load_generation_online(prev_start, prev_end)
        prev_prices = load_prices_online(prev_start, prev_end)
        prev_revenue = compute_revenue(prev_gen, prev_prices)
        previous = summary_numbers(compute_kpis(prev_gen), prev_revenue)
        prev_monthly = monthly_table(prev_gen, prev_prices)
    except Exception as error:
        previous = None  # ohne Vorjahresdaten zeigen wir keinen Vergleich, aber den Grund
        prev_monthly = None
        prev_error = f"Grund: {error}"
    else:
        prev_error = ""
    current = summary_numbers(kpis, revenue)


    # Titelseite: Kennzahlen, Anteil erneuerbarer Energie und Tabelle stehen über den Reitern
    show_key_figures(kpis, current, previous)
    show_year_comparison(current, previous, prev_start, prev_end, prev_error)

    tab_generation, tab_revenue, tab_prices, tab_variance, tab_report = st.tabs(
        ["Erzeugung", "Erlöse", "Preise und Risiko", "Abweichungen", "Bericht"]
    )
    with tab_generation:
        show_start(kpis)
    with tab_revenue:
        show_revenue(revenue, monthly_table(df, prices))
    with tab_prices:
        show_prices(prices, df)
    with tab_variance:
        show_forecast(df)
        st.divider()
        plan_variance(monthly_table(df, prices), prev_monthly)
    with tab_report:
        show_report(kpis, revenue, source, api_key)

    st.divider()
    st.markdown(
        "**Datenquelle:** Energy‑Charts (Fraunhofer ISE). Die Strompreise stammen von der "
        "Bundesnetzagentur (SMARD), Lizenz CC BY 4.0."
    )
    st.caption(
        f"Zeitraum: {kpis['period'][0]:%d.%m.%Y} bis {kpis['period'][1]:%d.%m.%Y}. "
        f"Datenstand: vollständige Daten bis {kpis['period'][1]:%d.%m.%Y %H:%M}. "
        "Die neuesten Werte können später noch korrigiert werden."
    )


if __name__ == "__main__":
    main()
