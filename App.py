import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Stromspeicher-Analyse", layout="wide")
st.title("⚡ Stromspeicherbedarf Deutschland – Fourier-Analyse")
st.markdown("Methode: Residuum (Verbrauch − EE) → FFT → Speicherkapazität = $A / (\\pi f)$ pro Frequenzkomponente")

# ── Upload im Hauptbereich ────────────────────────────────────────────────────
with st.expander("📂 Datei-Upload (SMARD-CSV)", expanded=True):
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        verbrauch_file = st.file_uploader("Verbrauch CSV", type="csv", key="v")
    with col_u2:
        erzeugung_file = st.file_uploader("Erzeugung CSV", type="csv", key="e")
    st.caption("Dateien von [smard.de](https://www.smard.de/home/downloadcenter/download-marktdaten/) · Format: Viertelstunde, Deutschland")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ EE-Kapazitätsfaktoren")
    wind_on_f  = st.slider("Wind Onshore ×",  0.0, 5.0, 1.0, 0.1)
    wind_off_f = st.slider("Wind Offshore ×", 0.0, 5.0, 1.0, 0.1)
    solar_f    = st.slider("Photovoltaik ×",  0.0, 5.0, 1.0, 0.1)
    bio_f      = st.slider("Biomasse ×",      0.0, 3.0, 1.0, 0.1)
    wasser_f   = st.slider("Wasserkraft ×",   0.0, 3.0, 1.0, 0.1)

    st.markdown("---")
    st.markdown("**Referenz:** Alle deutschen Pumpspeicher ≈ 40 GWh")


# ── Hilfsfunktionen ───────────────────────────────────────────────────────────
@st.cache_data
def lade_daten(verbrauch_bytes, erzeugung_bytes):
    import io
    ee_cols = ['Biomasse [MWh] Originalauflösungen',
               'Wasserkraft [MWh] Originalauflösungen',
               'Wind Offshore [MWh] Originalauflösungen',
               'Wind Onshore [MWh] Originalauflösungen',
               'Photovoltaik [MWh] Originalauflösungen',
               'Sonstige Erneuerbare [MWh] Originalauflösungen']

    verbrauch = pd.read_csv(io.BytesIO(verbrauch_bytes), sep=';', decimal=',',
                             thousands='.', parse_dates=['Datum von'], dayfirst=True)
    erzeugung = pd.read_csv(io.BytesIO(erzeugung_bytes), sep=';', decimal=',',
                             thousands='.', parse_dates=['Datum von'], dayfirst=True)

    df = verbrauch[['Datum von', 'Netzlast [MWh] Originalauflösungen']].copy()
    df.columns = ['ts', 'verbrauch']
    df = df.set_index('ts')

    ee = erzeugung[['Datum von'] + ee_cols].set_index('Datum von')
    for c in ee_cols:
        ee[c] = pd.to_numeric(ee[c], errors='coerce')

    df['Biomasse']          = ee['Biomasse [MWh] Originalauflösungen']
    df['Wasserkraft']       = ee['Wasserkraft [MWh] Originalauflösungen']
    df['Wind Offshore']     = ee['Wind Offshore [MWh] Originalauflösungen']
    df['Wind Onshore']      = ee['Wind Onshore [MWh] Originalauflösungen']
    df['Photovoltaik']      = ee['Photovoltaik [MWh] Originalauflösungen']
    df['Sonstige Erneuerbare'] = ee['Sonstige Erneuerbare [MWh] Originalauflösungen']

    return df.dropna().sort_index()


def berechne_speicher(df, wind_off_f, wind_on_f, solar_f, bio_f, wasser_f):
    ee = (df['Biomasse']          * bio_f   +
          df['Wasserkraft']       * wasser_f +
          df['Wind Offshore']     * wind_off_f +
          df['Wind Onshore']      * wind_on_f  +
          df['Photovoltaik']      * solar_f    +
          df['Sonstige Erneuerbare'])

    residuum = df['verbrauch'] - ee
    signal_ac = residuum.values - residuum.mean()

    N = len(signal_ac)
    dt = 0.25  # Stunden
    fft_vals   = np.fft.rfft(signal_ac)
    freqs      = np.fft.rfftfreq(N, d=dt)
    amplitudes = np.abs(fft_vals) * 2 / N

    with np.errstate(divide='ignore', invalid='ignore'):
        speicher_mwh = np.where(freqs > 0, amplitudes / (np.pi * freqs), 0)

    perioden_h = np.where(freqs > 0, 1.0 / freqs, np.inf)

    labels = ['< 1h', '1–6h', '6–24h', '1–7 Tage', '1–4 Wochen', '1–6 Monate', '> 6 Monate']
    bins   = [0, 1, 6, 24, 24*7, 24*28, 24*180, np.inf]

    result = {}
    for i in range(len(bins) - 1):
        mask = (perioden_h >= bins[i]) & (perioden_h < bins[i+1])
        result[labels[i]] = speicher_mwh[mask].sum() / 1e3  # GWh

    return result, residuum, ee, freqs, speicher_mwh, perioden_h


# ── Hauptbereich ──────────────────────────────────────────────────────────────
import os

FALLBACK_VERBRAUCH = os.path.join(os.path.dirname(__file__), "Realisierter_Stromverbrauch_2025.csv")
FALLBACK_ERZEUGUNG = os.path.join(os.path.dirname(__file__), "Realisierte_Erzeugung_2025.csv")

def _fallback_verfuegbar():
    return (os.path.exists(FALLBACK_VERBRAUCH) and os.path.getsize(FALLBACK_VERBRAUCH) > 10 and
            os.path.exists(FALLBACK_ERZEUGUNG) and os.path.getsize(FALLBACK_ERZEUGUNG) > 10)

if verbrauch_file and erzeugung_file:
    verbrauch_bytes = verbrauch_file.read()
    erzeugung_bytes = erzeugung_file.read()
elif _fallback_verfuegbar():
    st.info("ℹ️ Keine Dateien hochgeladen – lokale Fallback-CSVs werden verwendet.")
    with open(FALLBACK_VERBRAUCH, "rb") as f:
        verbrauch_bytes = f.read()
    with open(FALLBACK_ERZEUGUNG, "rb") as f:
        erzeugung_bytes = f.read()
else:
    verbrauch_bytes = None
    erzeugung_bytes = None

if verbrauch_bytes and erzeugung_bytes:
    df = lade_daten(verbrauch_bytes, erzeugung_bytes)

    # Zeitraum-Auswahl
    st.subheader("📅 Zeitraum")
    col1, col2 = st.columns(2)
    min_date = df.index.min().date()
    max_date = df.index.max().date()
    with col1:
        start = st.date_input("Von", min_date, min_value=min_date, max_value=max_date)
    with col2:
        end   = st.date_input("Bis", max_date, min_value=min_date, max_value=max_date)

    df_sel = df.loc[str(start):str(end)]

    if len(df_sel) < 100:
        st.warning("Zu wenig Datenpunkte – bitte längeren Zeitraum wählen.")
        st.stop()

    # Berechnung
    result, residuum, ee, freqs, speicher_mwh, perioden_h = berechne_speicher(
        df_sel, wind_off_f, wind_on_f, solar_f, bio_f, wasser_f)

    gesamt = sum(result.values())
    ee_anteil = ee.mean() / df_sel['verbrauch'].mean() * 100

    # ── KPIs ──────────────────────────────────────────────────────────────────
    st.subheader("📊 Kennzahlen")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Gesamtspeicherbedarf", f"{gesamt:.0f} GWh")
    k2.metric("EE-Deckungsgrad (Mittel)", f"{ee_anteil:.1f} %")
    k3.metric("Residuum Mittelwert", f"{residuum.mean()/1e3:.1f} GWh/h")
    k4.metric("Faktor vs. Pumpspeicher (~40 GWh)", f"{gesamt/40:.0f}×")

    # ── Plot 1: Balkendiagramm ─────────────────────────────────────────────────
    st.subheader("🔋 Speicherbedarf nach Zeitskala")
    labels = list(result.keys())
    values = list(result.values())

    colors = ['#2ecc71', '#27ae60', '#f39c12', '#e67e22', '#e74c3c', '#c0392b', '#8e44ad']
    fig1 = go.Figure(go.Bar(
        x=labels, y=values, marker_color=colors,
        text=[f"{v:.0f} GWh" for v in values],
        textposition='outside'
    ))
    fig1.add_hline(y=40, line_dash="dash", line_color="steelblue",
                   annotation_text="Alle Pumpspeicher DE (~40 GWh)",
                   annotation_position="top left")
    fig1.update_layout(
        yaxis_title="Speicherkapazität [GWh]",
        xaxis_title="Zeitskala",
        height=450,
        plot_bgcolor='white',
        yaxis=dict(gridcolor='lightgrey')
    )
    st.plotly_chart(fig1, use_container_width=True)

    # ── Plot 2: Zeitreihe + FFT-Spektrum ──────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("📈 Zeitreihe (Tagesmittel)")
        daily = df_sel[['verbrauch']].resample('D').mean()
        daily['ee'] = ee.resample('D').mean()
        daily['residuum'] = residuum.resample('D').mean()

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=daily.index, y=daily['verbrauch'],
                                   name='Verbrauch', line=dict(color='steelblue')))
        fig2.add_trace(go.Scatter(x=daily.index, y=daily['ee'],
                                   name='EE (skaliert)', line=dict(color='green')))
        fig2.add_trace(go.Scatter(x=daily.index, y=daily['residuum'],
                                   name='Residuum', line=dict(color='crimson')))
        fig2.add_hline(y=0, line_dash="dash", line_color="black", line_width=0.8)
        fig2.update_layout(yaxis_title="MWh", height=380,
                            plot_bgcolor='white', yaxis=dict(gridcolor='lightgrey'))
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        st.subheader("🌊 FFT-Spektrum (log-log)")
        mask = (freqs > 0) & (perioden_h < 400 * 24)
        perioden_tage = perioden_h[mask] / 24

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=perioden_tage, y=speicher_mwh[mask],
            mode='lines', line=dict(color='steelblue', width=0.8),
            name='Speicherbedarf'))

        for p, name in [(1, 'Tag'), (7, 'Woche'), (30, 'Monat'), (365, 'Jahr')]:
            fig3.add_vline(x=p, line_dash="dash", line_color="red", opacity=0.5,
                           annotation_text=name, annotation_position="top")

        fig3.update_layout(
            xaxis_type='log', yaxis_type='log',
            xaxis_title='Periode [Tage]', yaxis_title='Speicherkapazität [MWh]',
            height=380, plot_bgcolor='white',
            xaxis=dict(gridcolor='lightgrey'), yaxis=dict(gridcolor='lightgrey'))
        st.plotly_chart(fig3, use_container_width=True)

    # ── Rohdaten ──────────────────────────────────────────────────────────────
    with st.expander("📋 Rohdaten (Tagesmittel)"):
        st.dataframe(daily.round(1), use_container_width=True)

else:
    st.info("Bitte oben beide SMARD-CSV-Dateien hochladen – oder lokale Fallback-CSVs (`Realisierter_Stromverbrauch_2025.csv` / `Realisierte_Erzeugung_2025.csv`) im App-Verzeichnis bereitstellen.")
