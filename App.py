import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Stromspeicher-Analyse", layout="wide")
st.title("⚡ Stromspeicherbedarf Deutschland – Fourier-Analyse")

# ══════════════════════════════════════════════════════════════════════════════
# 📂 Abschnitt 1: Daten laden
# ══════════════════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════════════
# Hilfsfunktionen
# ══════════════════════════════════════════════════════════════════════════════
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

    df['Biomasse']             = ee['Biomasse [MWh] Originalauflösungen']
    df['Wasserkraft']          = ee['Wasserkraft [MWh] Originalauflösungen']
    df['Wind Offshore']        = ee['Wind Offshore [MWh] Originalauflösungen']
    df['Wind Onshore']         = ee['Wind Onshore [MWh] Originalauflösungen']
    df['Photovoltaik']         = ee['Photovoltaik [MWh] Originalauflösungen']
    df['Sonstige Erneuerbare'] = ee['Sonstige Erneuerbare [MWh] Originalauflösungen']

    return df.dropna().sort_index()


def _fft_bins(signal, dt=0.25):
    """FFT eines AC-Signals → Speicherkapazität und Leistung pro Zeitskalen-Bin."""
    N          = len(signal)
    fft_vals   = np.fft.rfft(signal)
    freqs      = np.fft.rfftfreq(N, d=dt)
    amplitudes = np.abs(fft_vals) * 2 / N

    with np.errstate(divide='ignore', invalid='ignore'):
        speicher_mwh = np.where(freqs > 0, amplitudes / (np.pi * freqs), 0)

    perioden_h  = np.where(freqs > 0, 1.0 / freqs, np.inf)
    leistung_mw = amplitudes * 2 * np.pi * freqs  # P_max = 2πf·A

    labels = ['< 1h', '1–6h', '6–24h', '1–7 Tage', '1–4 Wochen', '1–6 Monate', '> 6 Monate']
    bins   = [0, 1, 6, 24, 24*7, 24*28, 24*180, np.inf]

    res_e, res_l, res_p = {}, {}, {}
    for i in range(len(bins) - 1):
        mask             = (perioden_h >= bins[i]) & (perioden_h < bins[i+1])
        res_e[labels[i]] = speicher_mwh[mask].sum() / 1e3          # GWh
        res_l[labels[i]] = leistung_mw[mask].sum()  / 1e3          # GW (2πf·A)
        res_p[labels[i]] = (amplitudes[mask] * 4).sum() / 1e3      # GW (Ak/dt)

    return res_e, res_l, res_p, freqs, speicher_mwh, perioden_h


def berechne_speicher(df, wind_off_f, wind_on_f, solar_f, bio_f, wasser_f):
    ee = (df['Biomasse']          * bio_f      +
          df['Wasserkraft']       * wasser_f   +
          df['Wind Offshore']     * wind_off_f +
          df['Wind Onshore']      * wind_on_f  +
          df['Photovoltaik']      * solar_f    +
          df['Sonstige Erneuerbare'])

    residuum = df['verbrauch'] - ee
    dc_mwh   = residuum.mean()

    # ── Ohne Overbuild-Korrektur ──────────────────────────────────────────
    signal_ac = residuum.values - dc_mwh
    res_e, res_l, res_p, freqs, speicher_mwh, perioden_h = _fft_bins(signal_ac)

    # ── Mit Overbuild-Korrektur (Deadband-Logik) ──────────────────────────
    # Wenn DC < 0 (Überschuss), puffert der Überschuss Spitzen – aber nie über 0 hinaus.
    if dc_mwh < 0:
        residuum_korr = np.where(
            residuum > 0,
            np.maximum(residuum - abs(dc_mwh), 0),
            np.minimum(residuum + abs(dc_mwh), 0)
        )
    else:
        residuum_korr = residuum.values

    signal_ac_korr = residuum_korr - residuum_korr.mean()
    res_e_k, res_l_k, res_p_k, _, _, _ = _fft_bins(signal_ac_korr)

    return (res_e, res_l, res_p,
            res_e_k, res_l_k, res_p_k,
            residuum, ee,
            freqs, speicher_mwh, perioden_h, dc_mwh)


# Farben für die 7 Zeitskalen-Bins (konsistent in allen Diagrammen)
BIN_COLORS = ['#2ecc71', '#27ae60', '#f39c12', '#e67e22', '#e74c3c', '#c0392b', '#8e44ad']


def _ragone_fig(pts_vor, pts_nach=None):
    """Ragone-Diagramm mit Technologie-Rechtecken.
    pts_vor / pts_nach = (labels, x_gwh, y_gw)
    """
    fig = go.Figure()

    # (Name, x0, x1, y0, y1, fill-rgba, Rahmenfarbe)
    tech_boxes = [
        ('Kondensatoren',  1e-5,  0.01,   10,    1e4,  'rgba(127,140,141,0.15)', '#7f8c8d'),
        ('Li-Ionen',       0.001, 200,    0.05,  2000, 'rgba(52,152,219,0.18)',  '#2980b9'),
        ('Redox-Flow',     1,     5000,   0.001, 10,   'rgba(26,188,156,0.18)',  '#1abc9c'),
        ('Pumpspeicher',   0.5,   2000,   0.01,  40,   'rgba(46,204,113,0.18)',  '#27ae60'),
        ('CAES',           10,    10000,  0.005, 5,    'rgba(230,126,34,0.18)',  '#d35400'),
        ('Wasserstoff/PtX',100,   1e7,    0.001, 30,   'rgba(155,89,182,0.18)', '#8e44ad'),
    ]

    for name, x0, x1, y0, y1, fill, lc in tech_boxes:
        fig.add_shape(type='rect',
            x0=x0, x1=x1, y0=y0, y1=y1, xref='x', yref='y',
            fillcolor=fill, line=dict(color=lc, width=1), layer='below')
        fig.add_annotation(
            x=np.sqrt(x0 * x1), y=np.sqrt(y0 * y1),
            text=f"<b>{name}</b>", showarrow=False,
            font=dict(size=9, color=lc), xref='x', yref='y')

    fig.add_hrect(y0=30, y1=58, fillcolor='rgba(231,76,60,0.07)', line_width=0,
                  annotation_text='Gas DE (30–58 GW)',
                  annotation_position='top left', annotation_font_size=9)
    fig.add_hrect(y0=10, y1=22, fillcolor='rgba(52,73,94,0.07)', line_width=0,
                  annotation_text='Kohle DE (~15 GW)',
                  annotation_position='bottom left', annotation_font_size=9)

    labels_v, x_v, y_v = pts_vor
    fig.add_trace(go.Scatter(
        x=x_v, y=y_v, mode='markers+text',
        marker=dict(size=13, color=BIN_COLORS, line=dict(width=1.5, color='white')),
        text=labels_v, textposition='top center', textfont=dict(size=9),
        name='Ohne Overbuild',
        hovertemplate='<b>%{text}</b><br>%{x:.2g} GWh / %{y:.2g} GW<extra></extra>'
    ))

    if pts_nach:
        labels_n, x_n, y_n = pts_nach
        fig.add_trace(go.Scatter(
            x=x_n, y=y_n, mode='markers+text',
            marker=dict(size=10, color=BIN_COLORS, symbol='diamond',
                        line=dict(width=1.5, color='white')),
            text=labels_n, textposition='bottom center', textfont=dict(size=9),
            name='Nach Overbuild',
            hovertemplate='<b>%{text}</b><br>%{x:.2g} GWh / %{y:.2g} GW<extra></extra>'
        ))

    fig.update_layout(
        xaxis_type='log', yaxis_type='log',
        xaxis_title='Speicherkapazität [GWh]',
        yaxis_title='Spitzenleistung [GW]',
        height=540, plot_bgcolor='white',
        xaxis=dict(gridcolor='lightgrey', range=[-4, 7]),
        yaxis=dict(gridcolor='lightgrey', range=[-3, 4]),
        showlegend=pts_nach is not None,
        legend=dict(x=0.01, y=0.99)
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Daten-Routing: Upload → Fallback → Hinweis
# ══════════════════════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════════════════════
# Hauptbereich (nur wenn Daten vorhanden)
# ══════════════════════════════════════════════════════════════════════════════
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

    # ── Alle Berechnungen auf einmal ──────────────────────────────────────
    (res_e, res_l, res_p,
     res_e_k, res_l_k, res_p_k,
     residuum, ee,
     freqs, speicher_mwh, perioden_h, dc_mwh) = berechne_speicher(
        df_sel, wind_off_f, wind_on_f, solar_f, bio_f, wasser_f)

    labels      = list(res_e.keys())
    gesamt      = sum(res_e.values())
    gesamt_k    = sum(res_e_k.values())
    ee_anteil   = ee.mean() / df_sel['verbrauch'].mean() * 100
    dc_gwh_year = dc_mwh * 8760 / 1e3   # MW (Mittelleistung) × 8760 h → GWh/Jahr
    curtailment = abs(min(dc_gwh_year, 0))

    # Tagesmittelwerte für Zeitreihen-Plot
    daily              = df_sel[['verbrauch']].resample('D').mean()
    daily['ee']        = ee.resample('D').mean()
    daily['residuum']  = residuum.resample('D').mean()

    # ══════════════════════════════════════════════════════════════════════
    # ⚙️ Abschnitt 2: EE-Skalierung
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.header("⚙️ EE-Skalierung")

    kpi1, kpi2 = st.columns(2)
    kpi1.metric("EE-Deckungsgrad (Mittel)", f"{ee_anteil:.1f} %",
                help="Mittlere EE-Einspeisung / Mittlerer Verbrauch")
    if dc_gwh_year > 0:
        kpi2.metric("DC-Komponente", f"+{dc_gwh_year:,.0f} GWh/Jahr",
                    help="Mittleres Defizit – muss durch Importe oder Konventionelle gedeckt werden")
    else:
        kpi2.metric("DC-Komponente", f"{dc_gwh_year:,.0f} GWh/Jahr",
                    help="Mittlerer Überschuss – wird exportiert oder curtailed")

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=daily.index, y=daily['verbrauch'],
                                name='Verbrauch', line=dict(color='steelblue')))
    fig_ts.add_trace(go.Scatter(x=daily.index, y=daily['ee'],
                                name='EE (skaliert)', line=dict(color='green')))
    fig_ts.add_trace(go.Scatter(x=daily.index, y=daily['residuum'],
                                name='Residuallast', line=dict(color='crimson')))
    fig_ts.add_hline(y=0, line_dash="dash", line_color="black", line_width=0.8)
    fig_ts.add_hline(y=dc_mwh, line_dash="dot", line_color="orange", line_width=1.2,
                     annotation_text=f"DC ({dc_gwh_year:+,.0f} GWh/J)",
                     annotation_position="bottom right")
    fig_ts.update_layout(yaxis_title="MWh", height=380, plot_bgcolor='white',
                         yaxis=dict(gridcolor='lightgrey'))
    st.plotly_chart(fig_ts, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    # 🌊 Schritt 1: Residuallast & FFT
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.header("🌊 Schritt 1 – Residuallast & FFT")
    st.markdown(
        "Die **Residuallast** ist die Differenz aus Verbrauch und erneuerbarer Einspeisung: "
        "$r(t) = V(t) - EE(t)$. "
        "Positive Werte bedeuten Defizit (Speicher oder konventionelle Kraftwerke nötig), "
        "negative Werte Überschuss. "
        "Eine Fourier-Transformation zerlegt dieses Signal in Frequenzkomponenten – "
        "und zeigt, auf welchen Zeitskalen die größten Schwankungen auftreten."
    )

    with st.expander("📐 Rechenweg"):
        st.markdown(r"""
**Signalmodell** – eine einzelne Frequenzkomponente:

$$E(t) = A \sin(2\pi f t)$$

**Speicherkapazität** (maximale Energiemenge, die aufgenommen oder abgegeben werden muss):

$$E_{\text{speicher}} = \frac{A}{\pi f}$$

**Maximale Lade-/Entladeleistung** (Ableitung des Energiesignals):

$$P_{\max} = 2\pi f \cdot A$$

Dabei ist $A$ die FFT-Amplitude [MWh] und $f$ die Frequenz [1/h].
Die Gesamtwerte pro Zeitskalen-Bin entstehen durch Summation aller enthaltenen Frequenzkomponenten.
        """)

    # FFT-Spektrum (log-log)
    mask    = (freqs > 0) & (perioden_h >= 6) & (perioden_h <= 400 * 24)
    pd_tage = perioden_h[mask] / 24
    sp_mwh  = speicher_mwh[mask]
    order   = np.argsort(pd_tage)
    pd_tage, sp_mwh = pd_tage[order], sp_mwh[order]

    fig_fft = go.Figure()
    fig_fft.add_trace(go.Scatter(
        x=pd_tage, y=sp_mwh,
        mode='lines', line=dict(color='steelblue', width=0.8),
        name='Speicherkapazität'))

    for p, name in [(1, 'Tag'), (7, 'Woche'), (30, 'Monat'), (365, 'Jahr')]:
        fig_fft.add_vline(x=p, line_dash="dash", line_color="red", opacity=0.5,
                          annotation_text=name, annotation_position="top")

    tick_vals = [6/24, 1, 7, 30, 182, 365]
    tick_text = ['6h', '1 Tag', '1 Wo.', '1 Mo.', '6 Mo.', '1 Jahr']
    fig_fft.update_layout(
        xaxis_type='log', yaxis_type='log',
        xaxis_title='Periode', yaxis_title='Speicherkapazität [MWh]',
        height=380, plot_bgcolor='white',
        xaxis=dict(gridcolor='lightgrey', tickvals=tick_vals, ticktext=tick_text,
                   range=[np.log10(6/24), np.log10(400)]),
        yaxis=dict(gridcolor='lightgrey'))
    st.plotly_chart(fig_fft, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    # 🔋 Schritt 2: Speicherbedarf (ohne Overbuild)
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.header("🔋 Schritt 2 – Speicherbedarf (ohne Overbuild)")

    k1, k2, k3 = st.columns(3)
    k1.metric("Gesamtspeicherbedarf", f"{gesamt:.0f} GWh")
    k2.metric("Faktor vs. Pumpspeicher (~40 GWh)", f"{gesamt/40:.0f}×")
    k3.metric("EE-Deckungsgrad", f"{ee_anteil:.1f} %")

    # Balkendiagramm: Speicherkapazität
    st.subheader("Speicherkapazität nach Zeitskala")
    fig_e = go.Figure(go.Bar(
        x=labels, y=list(res_e.values()), marker_color=BIN_COLORS,
        text=[f"{v:.0f} GWh" for v in res_e.values()],
        textposition='outside'
    ))
    fig_e.add_hline(y=40, line_dash="dash", line_color="steelblue",
                    annotation_text="Alle Pumpspeicher DE (~40 GWh)",
                    annotation_position="top left")
    fig_e.update_layout(
        yaxis_title="Speicherkapazität [GWh]", xaxis_title="Zeitskala",
        height=430, plot_bgcolor='white', yaxis=dict(gridcolor='lightgrey'))
    st.plotly_chart(fig_e, use_container_width=True)

    # Balkendiagramm: Spitzenleistung
    st.subheader("Maximale Speicherleistung nach Zeitskala")
    fig_lw = go.Figure(go.Bar(
        x=labels, y=list(res_l.values()), marker_color=BIN_COLORS,
        text=[f"{v:.1f} GW" for v in res_l.values()],
        textposition='outside'
    ))
    fig_lw.update_layout(
        yaxis_title="Leistung [GW]", xaxis_title="Zeitskala",
        height=400, plot_bgcolor='white',
        yaxis=dict(gridcolor='lightgrey'),
        annotations=[dict(
            text="P<sub>max</sub> = 2πf · A &nbsp;(Ableitung von E = A · sin(2πft))",
            xref='paper', yref='paper', x=1.0, y=1.02,
            xanchor='right', yanchor='bottom',
            showarrow=False, font=dict(size=10, color='grey')
        )]
    )
    st.plotly_chart(fig_lw, use_container_width=True)

    # Ragone-Diagramm (ohne Overbuild)
    st.subheader("📍 Ragone-Diagramm: Speicherkapazität vs. Spitzenleistung")
    pts_vor = (labels,
               [res_e[l] for l in labels],
               [res_p[l] for l in labels])
    st.plotly_chart(_ragone_fig(pts_vor), use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    # 🏗️ Schritt 3: Overbuild & Curtailment
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.header("🏗️ Schritt 3 – Overbuild & Curtailment")
    st.markdown(
        "Wenn mehr EE installiert ist als der mittlere Verbrauch (**Overbuild**), "
        "entsteht eine negative DC-Komponente im Residuum. "
        "Dieser permanente Überschuss kann positive Lastspitzen abpuffern – "
        "allerdings nur solange das Residuum über null liegt (**Deadband**). "
        "Negative Restmengen lassen sich nicht weiter reduzieren, "
        "da man Energie nicht 'zurücknehmen' kann. "
        "Der nicht nutzbare Teil wird als **Curtailment** (Abregelung) verbucht."
    )

    if dc_mwh < 0:
        st.success(
            f"✅ Overbuild aktiv: DC = {dc_gwh_year:,.0f} GWh/Jahr "
            f"→ Curtailment ≈ {curtailment:,.0f} GWh/Jahr"
        )
    else:
        st.info("ℹ️ Kein Overbuild (DC ≥ 0) – Korrektur hat keinen Effekt auf das Spektrum.")

    kc1, kc2, kc3 = st.columns(3)
    kc1.metric("DC-Komponente", f"{dc_gwh_year:+,.0f} GWh/Jahr")
    kc2.metric("Curtailment (bei DC < 0)", f"{curtailment:,.0f} GWh/Jahr")
    kc3.metric("Speicherbedarf nach Korrektur", f"{gesamt_k:.0f} GWh",
               delta=f"{gesamt_k - gesamt:+.0f} GWh")

    # Vergleichs-Balken: Speicherkapazität
    st.subheader("Vergleich Speicherkapazität: vor vs. nach Overbuild-Korrektur")
    fig_vgl_e = go.Figure()
    fig_vgl_e.add_trace(go.Bar(
        name='Ohne Korrektur', x=labels, y=list(res_e.values()),
        marker_color='steelblue',
        text=[f"{v:.0f}" for v in res_e.values()], textposition='outside'
    ))
    fig_vgl_e.add_trace(go.Bar(
        name='Nach Korrektur', x=labels, y=list(res_e_k.values()),
        marker_color='tomato',
        text=[f"{v:.0f}" for v in res_e_k.values()], textposition='outside'
    ))
    fig_vgl_e.add_hline(y=40, line_dash="dash", line_color="grey",
                        annotation_text="Pumpspeicher DE (~40 GWh)",
                        annotation_position="top left")
    fig_vgl_e.update_layout(
        barmode='group', yaxis_title="Speicherkapazität [GWh]",
        height=430, plot_bgcolor='white', yaxis=dict(gridcolor='lightgrey'))
    st.plotly_chart(fig_vgl_e, use_container_width=True)

    # Vergleichs-Balken: Spitzenleistung
    st.subheader("Vergleich Spitzenleistung: vor vs. nach Overbuild-Korrektur")
    fig_vgl_l = go.Figure()
    fig_vgl_l.add_trace(go.Bar(
        name='Ohne Korrektur', x=labels, y=list(res_l.values()),
        marker_color='steelblue',
        text=[f"{v:.1f}" for v in res_l.values()], textposition='outside'
    ))
    fig_vgl_l.add_trace(go.Bar(
        name='Nach Korrektur', x=labels, y=list(res_l_k.values()),
        marker_color='tomato',
        text=[f"{v:.1f}" for v in res_l_k.values()], textposition='outside'
    ))
    fig_vgl_l.update_layout(
        barmode='group', yaxis_title="Spitzenleistung [GW]",
        height=430, plot_bgcolor='white', yaxis=dict(gridcolor='lightgrey'))
    st.plotly_chart(fig_vgl_l, use_container_width=True)

    # Ragone-Diagramm: vor vs. nach Overbuild
    st.subheader("📍 Ragone-Diagramm: vor vs. nach Overbuild-Korrektur")
    pts_nach = (labels,
                [res_e_k[l] for l in labels],
                [res_p_k[l] for l in labels])
    st.plotly_chart(_ragone_fig(pts_vor, pts_nach), use_container_width=True)

    # ── Rohdaten ──────────────────────────────────────────────────────────
    with st.expander("📋 Rohdaten (Tagesmittel)"):
        st.dataframe(daily.round(1), use_container_width=True)

else:
    st.info(
        "Bitte oben beide SMARD-CSV-Dateien hochladen – oder lokale Fallback-CSVs "
        "(`Realisierter_Stromverbrauch_2025.csv` / `Realisierte_Erzeugung_2025.csv`) "
        "im App-Verzeichnis bereitstellen."
    )
