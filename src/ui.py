import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta




class UI:

    def __init__(self, weather_data, sensor_data, inverter_data, wind_data, simulated_data) -> None:
        self.weather_data   = weather_data
        self.sensor_data    = sensor_data
        self.inverter_data  = inverter_data
        self.wind_data      = wind_data
        self.simulated_data = simulated_data
        # ------------------------------------------------------------
        # Page configuration
        # ------------------------------------------------------------
        st.set_page_config(
            page_title="Digital Tvilling: NTNU Gløshaugen",
            page_icon="☁️",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        self._load_styles()

    def render(self):
        self._render_sidebar()
        self._render_main()
        self._render_summary()

    def _load_styles(self):

        # ------------------------------------------------------------
        # Styling
        # ------------------------------------------------------------
        st.markdown("""
        <style>
            /* Overall app */
            .stApp {
                background: #0e0f13;
                color: #f2f2f4;
            }

            /* Hide Streamlit chrome */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}

            /* Sidebar */
            section[data-testid="stSidebar"] {
                background: #262731;
                border-right: 1px solid #30313b;
            }

            section[data-testid="stSidebar"] > div {
                padding-top: 1.0rem;
            }

            .sidebar-title {
                font-size: 16px;
                font-weight: 700;
                margin-bottom: 18px;
            }

            .sidebar-section {
                margin-top: 8px;
                padding: 0 0 22px 0;
                border-bottom: 1px solid #3a3b45;
            }

            .sidebar-label {
                color: #9697a2;
                font-size: 11px;
                margin: 0 0 7px 0;
            }

            .sidebar-value {
                color: #f5f5f7;
                font-size: 27px;
                font-weight: 400;
                line-height: 1.1;
                margin-bottom: 7px;
            }

            .sidebar-source {
                color: #a3a4ae;
                font-size: 10px;
                line-height: 1.35;
            }

            .weather-icon {
                font-size: 13px;
                margin-right: 5px;
            }

            /* Main area */
            .main-wrap {
                padding: 28px 42px 40px 42px;
            }

            .page-title {
                font-size: 31px;
                font-weight: 700;
                letter-spacing: -0.8px;
                margin: 0 0 5px 0;
            }

            .page-link {
                color: #a3a4ad;
                font-size: 14px;
                margin-left: 6px;
            }

            .tabs {
                display: flex;
                gap: 18px;
                border-bottom: 1px solid #30313a;
                margin: 23px 0 25px 0;
                height: 34px;
            }

            .tab-active {
                color: #ff4b4b;
                font-size: 13px;
                font-weight: 600;
                border-bottom: 2px solid #ff4b4b;
                padding-bottom: 11px;
            }

            .tab {
                color: #a5a6af;
                font-size: 13px;
                padding-bottom: 11px;
            }

            .section-title {
                font-size: 20px;
                font-weight: 650;
                margin: 0 0 24px 0;
            }

            .metric-label {
                color: #9a9ba5;
                font-size: 11px;
                font-weight: 600;
                margin-bottom: 5px;
            }

            .metric-value {
                color: #f4f4f5;
                font-size: 25px;
                font-weight: 400;
                line-height: 1.15;
                margin-bottom: 25px;
            }

            .control-label {
                color: #a7a8b1;
                font-size: 11px;
                font-weight: 600;
                margin-bottom: 3px;
            }

            .summary-title {
                color: #b0b1b9;
                font-size: 12px;
                font-weight: 600;
                margin: 12px 0 24px 0;
            }

            .summary-label {
                color: #a3a4ad;
                font-size: 11px;
                font-weight: 600;
                margin-bottom: 5px;
            }

            .summary-value {
                color: #f5f5f6;
                font-size: 25px;
                font-weight: 400;
            }

            /* Streamlit radio buttons */
            div[data-testid="stRadio"] > label {
                color: #a8a9b2 !important;
                font-size: 11px !important;
                font-weight: 600 !important;
            }

            div[data-testid="stRadio"] div[role="radiogroup"] {
                gap: 13px;
            }

            div[data-testid="stRadio"] div[role="radiogroup"] label {
                color: #e4e4e7 !important;
                font-size: 11px !important;
                font-weight: 400 !important;
            }

            /* Button */
            .stButton > button {
                width: 100%;
                height: 34px;
                border-radius: 7px;
                border: 1px solid #464752;
                background: #292a33;
                color: #dedee2;
                font-size: 11px;
            }

            .stButton > button:hover {
                border-color: #666873;
                color: #fff;
            }

            /* Reduce default gaps */
            div.block-container {
                padding-top: 0;
                padding-bottom: 0;
            }

            [data-testid="column"] {
                padding-top: 0 !important;
            }
        </style>
        """, unsafe_allow_html=True)

    def _weather_code_to_text(self, symbol_code):
        if not symbol_code: 
            return '🌤️ Mangler data'
        
        #splitter strengen ved '_' så det blir enklere å matche med symbolet
        hovedkode = symbol_code.split('_')[0]
        
        ikoner = {
            'clearsky': '☀️ Klart',
            'fair': '🌤️ Lettskyet',
            'partlycloudy': '⛅ Delvis skyet',
            'cloudy': '☁️ Overskyet',
            'lightrain': '🌦️ Lett regn',
            'rain': '🌧️ Regn',
            'heavyrain': '🌧️ Kraftig regn',
            'rainshowers': '🌦️ Regnbyger',
            'snow': '❄️ Snø',
            'lightsnow': '🌨️ Lett snø',
            'heavysnow': '❄️ Kraftig snø',
            'sleet': '🌧️❄️ Sludd',
            'fog': '🌫️ Tåke',
            'thunder': '⛈️ Tordenvær'
        }
        
        #Returnerer ikon som hører til symbolkoden
        return ikoner.get(hovedkode, '🌤️ Varierende')
    
    def _render_sidebar(self):

        # ------------------------------------------------------------
        # Sidebar
        # ------------------------------------------------------------
        with st.sidebar:
            st.markdown('<div class="sidebar-title">Været nå (Gløshaugen)</div>',
                        unsafe_allow_html=True)

            st.markdown('<div class="sidebar-label">Tidspunkt: 07.09.2026 15:00</div>',
                        unsafe_allow_html=True)

            st.markdown("""
            <div class="sidebar-section">
                <div style="font-size:13px;font-weight:650;margin:12px 0 18px 0;">
                    ☁️ Oversynet
                </div>
                <div class="sidebar-label">Vind</div>
                <div class="sidebar-value">1.1 m/s</div>
                <div class="sidebar-source">MET.no - Frost API, stasjon Voll (vind)</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="sidebar-section">
                <div class="sidebar-title" style="margin-top:18px;">
                    Sensormålinger (Taket)
                </div>
                <div class="sidebar-label">Tidspunkt: 21.05.2026 22:48:20</div>

                <div style="margin-top:22px;">
                    <div class="sidebar-label">Pyranometer</div>
                    <div class="sidebar-value">0 W/m²</div>
                </div>

                <div style="margin-top:18px;">
                    <div class="sidebar-label">Modultemperatur ⓘ</div>
                    <div class="sidebar-value">12.3 °C</div>
                </div>

                <div style="margin-top:18px;">
                    <div class="sidebar-label">Omgivelsestemperatur</div>
                    <div class="sidebar-value">12.5 °C</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
            st.markdown('<div class="sidebar-title">Visning</div>',
                        unsafe_allow_html=True)

            if st.button("Skjermmodus"):
                st.session_state["dark_mode"] = not st.session_state.get("dark_mode", False)

    def _render_main(self):

        # ------------------------------------------------------------
        # Main content
        # ------------------------------------------------------------
        st.markdown('<div class="main-wrap">', unsafe_allow_html=True)

        st.markdown(
            '<div class="page-title">Digital Tvilling: NTNU Gløshaugen '
            '<span class="page-link">↗</span></div>',
            unsafe_allow_html=True
        )

        st.markdown("""
        <div class="tabs">
            <div class="tab-active">Oversikt</div>
            <div class="tab">Detaljer &amp; Statistikk</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Sanntidsdata</div>',
                    unsafe_allow_html=True)

        # Metrics + chart controls
        left, right = st.columns([0.40, 0.60], gap="large")

        with left:
            m1, m2 = st.columns(2)

            with m1:
                st.markdown('<div class="metric-label">Simulert effekt</div>',
                            unsafe_allow_html=True)
                st.markdown('<div class="metric-value">0.00 kW</div>',
                            unsafe_allow_html=True)

            with m2:
                st.markdown('<div class="metric-label">Målt effekt</div>',
                            unsafe_allow_html=True)
                st.markdown('<div class="metric-value">1.54 kW</div>',
                            unsafe_allow_html=True)

            m3, m4 = st.columns(2)

            with m3:
                st.markdown('<div class="metric-label">Ytelsesratio ⓘ</div>',
                            unsafe_allow_html=True)
                st.markdown('<div class="metric-value">-</div>',
                            unsafe_allow_html=True)

            with m4:
                st.markdown('<div class="metric-label">Residual effekt</div>',
                            unsafe_allow_html=True)
                st.markdown('<div class="metric-value">-1.54 kW</div>',
                            unsafe_allow_html=True)

        with right:
            c1, c2 = st.columns([0.53, 0.47])

            with c1:
                st.markdown('<div class="control-label">Datatype</div>',
                            unsafe_allow_html=True)
                datatype = st.radio(
                    "Datatype",
                    ["Effekt (kW)", "Energi (kWh)"],
                    horizontal=True,
                    label_visibility="collapsed",
                    key="datatype",
                )

            with c2:
                st.markdown('<div class="control-label">Periode</div>',
                            unsafe_allow_html=True)
                period = st.radio(
                    "Periode",
                    ["24t", "7d", "30d", "År"],
                    horizontal=True,
                    label_visibility="collapsed",
                    key="period",
                )

            # Plot
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=self.sensor_data.index,
                y=self.sensor_data.values,
                mode="lines",
                name="Målt (SolarEdge)",
                line=dict(width=2),
                hovertemplate="%{x|%d.%m.%Y %H:%M}<br>Målt: %{y:.2f} kW<extra></extra>",
            ))

            fig.add_trace(go.Scatter(
                x=self.simulated_data.index,
                y=self.simulated_data.values,
                mode="lines",
                name="Simulert (Digital Tvilling)",
                line=dict(width=2),
                hovertemplate="%{x|%d.%m.%Y %H:%M}<br>Simulert: %{y:.2f} kW<extra></extra>",
            ))

            fig.update_layout(
                title=dict(
                    text="Faktisk vs Simulert Effektproduksjon",
                    font=dict(size=12, color="#d7d7db"),
                    x=0,
                    xanchor="left",
                ),
                height=300,
                margin=dict(l=48, r=12, t=43, b=38),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#a5a6af", size=10),
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.03,
                    xanchor="right",
                    x=1,
                    font=dict(size=10),
                ),
                xaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    tickfont=dict(size=9),
                    linecolor="#30313a",
                ),
                yaxis=dict(
                    title=dict(text="Effekt (kW)", font=dict(size=10)),
                    showgrid=True,
                    gridcolor="#282932",
                    zeroline=False,
                    tickfont=dict(size=9),
                    rangemode="tozero",
                ),
            )

            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    def _render_summary(self):
        # ------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------
        st.markdown(
            '<div style="height:1px;background:#30313a;margin:4px 0 0 0;"></div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="summary-title">Oppsummering for valgt periode (24t)</div>',
            unsafe_allow_html=True
        )

        s1, s2, s3, s4 = st.columns(4)

        with s1:
            st.markdown('<div class="summary-label">Totalt Målt Energi</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="summary-value">31.6 kWh</div>',
                        unsafe_allow_html=True)

        with s2:
            st.markdown('<div class="summary-label">Totalt Simulert Energi</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="summary-value">0.0 kWh</div>',
                        unsafe_allow_html=True)

        with s3:
            st.markdown('<div class="summary-label">Avvik</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="summary-value">31.6 kWh</div>',
                        unsafe_allow_html=True)

        with s4:
            st.markdown('<div class="summary-label">Ytelsesratio ⓘ</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="summary-value">0.0 %</div>',
                        unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)