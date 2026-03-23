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
    global_f = st.slider("🌐 Alle EE ×", 0.5, 3.0, 1.0, 0.05,
                          help="Skaliert alle nicht-gesperrten Quellen gleichzeitig")
    st.caption("🔒 = von globalem Slider ausgenommen")
    st.markdown("---")

    def _ee_row(label, key_lock, key_val, max_val=5.0):
        c1, c2 = st.columns([1, 5])
        locked = c1.checkbox("🔒", key=key_lock, help="Globalen Slider ignorieren")
        val    = c2.slider(label, 0.0, max_val, 1.0, 0.1, key=key_val)
        return val if locked else val * global_f

    wind_on_f  = _ee_row("Wind Onshore ×",  "lk_won",  "sv_won")
    wind_off_f = _ee_row("Wind Offshore ×", "lk_woff", "sv_woff")
    solar_f    = _ee_row("Photovoltaik ×",  "lk_sol",  "sv_sol")
    bio_f      = _ee_row("Biomasse ×",      "lk_bio",  "sv_bio", 3.0)
    wasser_f   = _ee_row("Wasserkraft ×",   "lk_was",  "sv_was", 3.0)

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

    # Leistung pro Frequenzkomponente: A · 2π · f (in MW, da A in MWh und f in 1/h)
    leistung_mw = amplitudes * 2 * np.pi * freqs

    labels = ['< 1h', '1–6h', '6–24h', '1–7 Tage', '1–4 Wochen', '1–6 Monate', '> 6 Monate']
    bins   = [0, 1, 6, 24, 24*7, 24*28, 24*180, np.inf]

    result          = {}
    result_leistung = {}
    result_pmax     = {}   # Spitzenleistung für Ragone: Pk = Ak / dt = Ak × 4
    for i in range(len(bins) - 1):
        mask = (perioden_h >= bins[i]) & (perioden_h < bins[i+1])
        result[labels[i]]          = speicher_mwh[mask].sum() / 1e3          # GWh
        result_leistung[labels[i]] = leistung_mw[mask].sum()  / 1e3          # GW (2πf)
        result_pmax[labels[i]]     = (amplitudes[mask] * 4).sum() / 1e3      # GW (Ak/dt)

    return result, result_leistung, result_pmax, residuum, ee, freqs, speicher_mwh, perioden_h, residuum.mean()


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
    result, result_leistung, result_pmax, residuum, ee, freqs, speicher_mwh, perioden_h, dc_mwh = berechne_speicher(
        df_sel, wind_off_f, wind_on_f, solar_f, bio_f, wasser_f)

    gesamt = sum(result.values())
    ee_anteil = ee.mean() / df_sel['verbrauch'].mean() * 100
    dc_gwh_year = dc_mwh * 8760 / 1e3  # MWh/h (Leistung) × 8760 h → GWh/Jahr

    # ── KPIs ──────────────────────────────────────────────────────────────────
    st.subheader("📊 Kennzahlen")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Gesamtspeicherbedarf", f"{gesamt:.0f} GWh")
    k2.metric("EE-Deckungsgrad (Mittel)", f"{ee_anteil:.1f} %")
    if dc_gwh_year > 0:
        dc_val_str  = f"+{dc_gwh_year:,.0f} GWh/Jahr"
        dc_help     = "➕ Mittleres Defizit – muss zugeführt werden (Importe / konventionell)"
    else:
        dc_val_str  = f"{dc_gwh_year:,.0f} GWh/Jahr"
        dc_help     = "➖ Mittlerer Überschuss – muss abgeführt werden (Export / Curtailment)"
    k3.metric("DC-Komponente", dc_val_str, help=dc_help)
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

    # ── Plot 1b: Leistungsdiagramm ────────────────────────────────────────────
    st.subheader("⚡ Maximale Speicherleistung pro Zeitskala")
    lw_labels = list(result_leistung.keys())
    lw_values = list(result_leistung.values())
    fig_lw = go.Figure(go.Bar(
        x=lw_labels, y=lw_values, marker_color=colors,
        text=[f"{v:.1f} GW" for v in lw_values],
        textposition='outside'
    ))
    fig_lw.update_layout(
        yaxis_title="Leistung [GW]",
        xaxis_title="Zeitskala",
        height=400,
        plot_bgcolor='white',
        yaxis=dict(gridcolor='lightgrey')
    )
    st.plotly_chart(fig_lw, use_container_width=True)

    # ── Ragone-Diagramm ────────────────────────────────────────────────────────
    st.subheader("📍 Ragone-Diagramm: Speicherkapazität vs. Spitzenleistung")

    fig_rag = go.Figure()

    # Technologie-Rechtecke (Daten-Koordinaten; Plotly übernimmt log-Transformation)
    tech_boxes = [
        # (Name,         x0,      x1,     y0,     y1,     fill-rgba,                    Rahmenfarbe)
        ('Schwungrad',   5e-5,    0.05,   0.5,    5000,   'rgba(149,165,166,0.18)',     '#7f8c8d'),
        ('Li-Ionen',     0.001,   200,    0.05,   2000,   'rgba(52,152,219,0.18)',      '#2980b9'),
        ('Pumpspeicher', 0.5,     2000,   0.01,   40,     'rgba(46,204,113,0.18)',      '#27ae60'),
        ('CAES',         10,      10000,  0.005,  5,      'rgba(230,126,34,0.18)',      '#d35400'),
        ('Wasserstoff',  100,     1e7,    0.001,  30,     'rgba(155,89,182,0.18)',      '#8e44ad'),
    ]

    for name, x0, x1, y0, y1, fill, lc in tech_boxes:
        fig_rag.add_shape(type='rect',
            x0=x0, x1=x1, y0=y0, y1=y1, xref='x', yref='y',
            fillcolor=fill, line=dict(color=lc, width=1), layer='below')
        fig_rag.add_annotation(
            x=np.sqrt(x0 * x1), y=np.sqrt(y0 * y1),  # geometrischer Mittelpunkt (log)
            text=f"<b>{name}</b>", showarrow=False,
            font=dict(size=9, color=lc), xref='x', yref='y')

    # Konventionelle Kraftwerke: horizontale Bänder
    fig_rag.add_hrect(y0=30, y1=58, fillcolor='rgba(231,76,60,0.07)', line_width=0,
                      annotation_text='Gas DE (30–58 GW)',
                      annotation_position='top left', annotation_font_size=9)
    fig_rag.add_hrect(y0=10, y1=22, fillcolor='rgba(52,73,94,0.07)', line_width=0,
                      annotation_text='Kohle DE (~15 GW)',
                      annotation_position='bottom left', annotation_font_size=9)

    # FFT-Punkte (ein Punkt pro Zeitskalen-Bin)
    x_pts = [result[l]      for l in labels]   # GWh
    y_pts = [result_pmax[l] for l in labels]   # GW

    fig_rag.add_trace(go.Scatter(
        x=x_pts, y=y_pts,
        mode='markers+text',
        marker=dict(size=13, color=colors, line=dict(width=1.5, color='white')),
        text=labels,
        textposition='top center',
        textfont=dict(size=9),
        name='FFT-Bins',
        hovertemplate='<b>%{text}</b><br>Kapazität: %{x:.2g} GWh<br>Leistung: %{y:.2g} GW<extra></extra>'
    ))

    fig_rag.update_layout(
        xaxis_type='log', yaxis_type='log',
        xaxis_title='Speicherkapazität [GWh]',
        yaxis_title='Spitzenleistung [GW]',
        height=540,
        plot_bgcolor='white',
        xaxis=dict(gridcolor='lightgrey', range=[-4, 7]),
        yaxis=dict(gridcolor='lightgrey', range=[-3, 4]),
        showlegend=False,
    )
    st.plotly_chart(fig_rag, use_container_width=True)

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
        fig2.add_hline(y=dc_mwh, line_dash="dot", line_color="orange", line_width=1.2,
                       annotation_text=f"DC ({dc_gwh_year:+,.0f} GWh/J)",
                       annotation_position="bottom right")
        fig2.update_layout(yaxis_title="MWh", height=380,
                            plot_bgcolor='white', yaxis=dict(gridcolor='lightgrey'))
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        st.subheader("🌊 FFT-Spektrum (log-log)")
        mask = (freqs > 0) & (perioden_h >= 6) & (perioden_h <= 400 * 24)
        pd_tage = perioden_h[mask] / 24
        sp_mwh  = speicher_mwh[mask]
        # Aufsteigend nach Periode sortieren (FFT liefert absteigende Reihenfolge)
        order = np.argsort(pd_tage)
        pd_tage, sp_mwh = pd_tage[order], sp_mwh[order]

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=pd_tage, y=sp_mwh,
            mode='lines', line=dict(color='steelblue', width=0.8),
            name='Speicherbedarf'))

        for p, name in [(1, 'Tag'), (7, 'Woche'), (30, 'Monat'), (365, 'Jahr')]:
            fig3.add_vline(x=p, line_dash="dash", line_color="red", opacity=0.5,
                           annotation_text=name, annotation_position="top")

        tick_vals = [6/24, 1, 7, 30, 182, 365]
        tick_text = ['6h', '1 Tag', '1 Wo.', '1 Mo.', '6 Mo.', '1 Jahr']
        fig3.update_layout(
            xaxis_type='log', yaxis_type='log',
            xaxis_title='Periode', yaxis_title='Speicherkapazität [MWh]',
            height=380, plot_bgcolor='white',
            xaxis=dict(gridcolor='lightgrey', tickvals=tick_vals, ticktext=tick_text,
                       range=[np.log10(6/24), np.log10(400)]),
            yaxis=dict(gridcolor='lightgrey'))
        st.plotly_chart(fig3, use_container_width=True)

    # ── Rohdaten ──────────────────────────────────────────────────────────────
    with st.expander("📋 Rohdaten (Tagesmittel)"):
        st.dataframe(daily.round(1), use_container_width=True)

else:
    st.info("Bitte oben beide SMARD-CSV-Dateien hochladen – oder lokale Fallback-CSVs (`Realisierter_Stromverbrauch_2025.csv` / `Realisierte_Erzeugung_2025.csv`) im App-Verzeichnis bereitstellen.")
