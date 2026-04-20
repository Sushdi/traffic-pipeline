import math
import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yaml
from datetime import datetime, timezone
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, f1_score
)
from sklearn.preprocessing import LabelBinarizer
from streamlit_folium import st_folium
# At the top of dashboard.py after imports
_API_KEY_ENV = os.environ.get("TOMTOM_API_KEY", "")

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Munich Traffic Intelligence",
    page_icon=":material/analytics:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background adjustments */
    .stApp {
        background-color: #F8FAFC !important;
        background-image:
            radial-gradient(circle at 16% 14%, rgba(34, 197, 94, 0.16), transparent 18%),
            radial-gradient(circle at 82% 18%, rgba(249, 115, 22, 0.12), transparent 20%),
            radial-gradient(circle at 50% 80%, rgba(239, 68, 68, 0.12), transparent 22%),
            linear-gradient(90deg, rgba(15, 23, 42, 0.06) 1px, transparent 1px),
            linear-gradient(0deg, rgba(15, 23, 42, 0.06) 1px, transparent 1px),
            radial-gradient(#CBD5E1 1px, transparent 1px);
        background-size:
            180px 180px,
            180px 180px,
            220px 220px,
            80px 80px,
            80px 80px,
            24px 24px;
        background-repeat: no-repeat, no-repeat, no-repeat, repeat, repeat, repeat;
        background-position: center top, center top, center bottom, center, center, center;
    }
    


    /* Metric Cards */
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        padding: 16px 24px;
        border-radius: 12px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease-in-out, box-shadow 0.2s;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    /* Primary Buttons */
    .stButton > button[kind="primary"] {
        background: #F5F5F5;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2);
    }
    .stButton > button[kind="primary"]:hover {
        background: #ced5e1;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
        border: none !important;
        color: white !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #FFFFFF;
        padding: 6px;
        border-radius: 10px;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        border: 1px solid #E2E8F0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        border-radius: 6px;
        padding: 0 20px;
        color: #64748B;
        font-weight: 500;
        background-color: transparent;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #F1F5F9 !important;
        color: #0F172A !important;
        font-weight: 600;
    }
    
    /* Metric styling */
    [data-testid="stMetricLabel"] {
        font-weight: 500;
        color: #64748B;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    [data-testid="stMetricDelta"] {
        font-size: 1rem;
        font-weight: 600;
    }

    /* Headers */
    h1, h2, h3, h4 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        color: #0F172A;
        letter-spacing: -0.5px;
    }
    
    [data-testid="stMarkdownContainer"] p {
        color: #475569;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }
    
</style>
""", unsafe_allow_html=True)

with open("params.yaml") as f:
    CFG = yaml.safe_load(f)

WAYPOINTS  = CFG["waypoints"]
MODEL_PATH = CFG["paths"]["model"]
PROC_PATH  = CFG["paths"]["processed"]
RAW_PATH   = CFG["paths"]["raw"]
EVAL_LOG_PATH = CFG["paths"].get("eval_log", "monitoring/eval_log.csv")
HORIZON_MINS  = (
    CFG["collection"].get("prediction_horizon_steps", 3)
    * CFG["collection"].get("interval_seconds", 300)
) // 60  # = 15

LOCATION_NAMES = [wp["name"].replace("_", " ").title() for wp in WAYPOINTS]
LOCATION_MAP   = {
    wp["name"].replace("_", " ").title(): wp
    for wp in WAYPOINTS
}

COLORS = {
    "free_flow": "#2ecc71",
    "moderate":  "#f39c12",
    "congested": "#e74c3c",
}

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_processed():
    return pd.read_csv(PROC_PATH, parse_dates=["timestamp"])

@st.cache_data
def load_raw():
    return pd.read_csv(RAW_PATH, parse_dates=["timestamp"])

bundle = load_model()
MODEL  = bundle["model"]
LE     = bundle["label_encoder"]
FEATS  = bundle["features"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## :material/directions_car: Munich Traffic")
    st.markdown("---")

    selected_location = st.selectbox(
        "Select area", LOCATION_NAMES, index=0
    )
    wp = LOCATION_MAP[selected_location]

    st.markdown("---")
    api_key = st.text_input(
        "TomTom API Key",
        value="",
        type="password",
        placeholder="Enter key (or set TOMTOM_API_KEY env var)",
        help="Required for live predictions tab."
    )
    # Use env var silently if input is empty
    api_key = api_key or _API_KEY_ENV
    st.markdown("---")
    st.markdown("### Model info")
    st.metric("Best model", bundle["best_model"].upper())
    st.metric("F1 score",   f"{bundle['best_f1']:.4f}")
    st.markdown("---")
    st.caption("Data: TomTom Flow API · Model: LightGBM")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    ":material/map: Live Traffic Map",
    ":material/analytics: Model Performance",
    ":material/search: Feature Analysis",
    ":material/history: Historical Trends",
    ":material/compare_arrows: Now vs Usual",
    ":material/monitoring: Prediction Accuracy & Drift",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Live Traffic Map
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Live Munich Traffic — Real-time predictions")
    st.markdown("""
<div style="position:relative; width:100%; height:180px; margin:16px 0 24px; border-radius:16px; overflow:hidden;">
  <svg width="100%" viewBox="0 0 680 180" xmlns="http://www.w3.org/2000/svg" style="position:absolute;inset:0;">
    <style>
      .road{fill:#94a3b8;opacity:.13}.lane{stroke:#cbd5e1;stroke-width:2;stroke-dasharray:22 14;fill:none;opacity:.25}
      .edge{stroke:#f1f5f9;stroke-width:1.5;fill:none;opacity:.2}.car-body{opacity:.12}
      .tl-pole{stroke:#94a3b8;stroke-width:2;fill:none;opacity:.18}.tl-box{fill:#475569;opacity:.12}
      .tl-red{fill:#ef4444;opacity:.22}.tl-amber{fill:#f59e0b;opacity:.22}.tl-green{fill:#22c55e;opacity:.22}
      .crosswalk{fill:#e2e8f0;opacity:.14}.sign{fill:#64748b;opacity:.1}
    </style>
    <!-- Main road -->
    <rect class="road" x="0" y="68" width="680" height="52"/>
    <line class="edge" x1="0" y1="68" x2="680" y2="68"/>
    <line class="edge" x1="0" y1="120" x2="680" y2="120"/>
    <line class="lane" x1="0" y1="94" x2="680" y2="94"/>
    <line x1="0" y1="79" x2="680" y2="79" stroke="#cbd5e1" stroke-width="1.2" stroke-dasharray="14 20" fill="none" opacity=".15"/>
    <line x1="0" y1="109" x2="680" y2="109" stroke="#cbd5e1" stroke-width="1.2" stroke-dasharray="14 20" fill="none" opacity=".15"/>
    <!-- Vertical road -->
    <rect class="road" x="268" y="0" width="48" height="180" style="opacity:.1"/>
    <line class="edge" x1="268" y1="0" x2="268" y2="180"/>
    <line class="edge" x1="316" y1="0" x2="316" y2="180"/>
    <line class="lane" x1="292" y1="0" x2="292" y2="60"/>
    <line class="lane" x1="292" y1="128" x2="292" y2="180"/>
    <!-- Crosswalk -->
    <rect class="crosswalk" x="316" y="72" width="8" height="44"/>
    <rect class="crosswalk" x="328" y="72" width="8" height="44"/>
    <rect class="crosswalk" x="340" y="72" width="8" height="44"/>
    <rect class="crosswalk" x="352" y="72" width="8" height="44"/>
    <rect class="crosswalk" x="364" y="72" width="8" height="44"/>
    <!-- Cars -->
    <g class="car-body"><rect x="60" y="74" width="48" height="20" rx="4" fill="#3b82f6"/><rect x="68" y="70" width="32" height="10" rx="3" fill="#3b82f6"/><circle cx="70" cy="94" r="4" fill="#1e293b"/><circle cx="98" cy="94" r="4" fill="#1e293b"/></g>
    <g class="car-body" transform="scale(-1,1) translate(-480,0)"><rect x="390" y="98" width="44" height="18" rx="4" fill="#64748b"/><rect x="398" y="94" width="28" height="10" rx="3" fill="#64748b"/><circle cx="398" cy="116" r="4" fill="#1e293b"/><circle cx="424" cy="116" r="4" fill="#1e293b"/></g>
    <g class="car-body"><rect x="520" y="75" width="42" height="18" rx="4" fill="#475569"/><rect x="528" y="71" width="26" height="10" rx="3" fill="#475569"/><circle cx="530" cy="93" r="4" fill="#1e293b"/><circle cx="552" cy="93" r="4" fill="#1e293b"/></g>
    <!-- Traffic lights -->
    <line class="tl-pole" x1="248" y1="68" x2="248" y2="120"/>
    <rect x="241" y="38" width="14" height="30" rx="2" class="tl-box"/>
    <circle cx="248" cy="44" r="4" class="tl-red"/><circle cx="248" cy="53" r="4" class="tl-amber"/><circle cx="248" cy="62" r="4" class="tl-green"/>
    <line class="tl-pole" x1="334" y1="68" x2="334" y2="40"/>
    <rect x="327" y="10" width="14" height="30" rx="2" class="tl-box"/>
    <circle cx="334" cy="16" r="4" class="tl-red"/><circle cx="334" cy="25" r="4" class="tl-amber"/><circle cx="334" cy="34" r="4" class="tl-green"/>
    <line class="tl-pole" x1="590" y1="120" x2="590" y2="68"/>
    <rect x="583" y="40" width="14" height="28" rx="2" class="tl-box"/>
    <circle cx="590" cy="46" r="4" class="tl-red"/><circle cx="590" cy="54" r="4" class="tl-amber"/><circle cx="590" cy="62" r="4" class="tl-green"/>
    <!-- Signs -->
    <line x1="30" y1="68" x2="30" y2="40" stroke="#94a3b8" stroke-width="1.5" opacity=".14"/>
    <rect class="sign" x="14" y="30" width="32" height="18" rx="3"/>
    <line x1="640" y1="68" x2="640" y2="44" stroke="#94a3b8" stroke-width="1.5" opacity=".14"/>
    <rect class="sign" x="624" y="34" width="32" height="18" rx="3"/>
  </svg>
</div>
""", unsafe_allow_html=True)
    def fetch_tomtom(lat, lon, key):
        url = (
            "https://api.tomtom.com/traffic/services/4"
            f"/flowSegmentData/absolute/10/json"
            f"?point={lat},{lon}&key={key}"
        )
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()["flowSegmentData"]

    def build_features(fsd):
        now    = datetime.now(timezone.utc)
        hour   = now.hour
        minute = now.minute
        dow    = now.weekday()
        cur_speed  = float(fsd["currentSpeed"])
        free_speed = float(fsd["freeFlowSpeed"])
        cur_tt     = float(fsd["currentTravelTime"])
        free_tt    = float(fsd["freeFlowTravelTime"])
        frc_map    = {"FRC0":0,"FRC1":1,"FRC2":2,"FRC3":3,
                      "FRC4":4,"FRC5":5,"FRC6":6,"FRC7":7}
        return {
            "hour_sin":  math.sin(2 * math.pi * hour / 24),
            "hour_cos":  math.cos(2 * math.pi * hour / 24),
            "dow_sin":   math.sin(2 * math.pi * dow / 7),
            "dow_cos":   math.cos(2 * math.pi * dow / 7),
            "min_sin":   math.sin(2 * math.pi * minute / 60),
            "min_cos":   math.cos(2 * math.pi * minute / 60),
            "is_weekend":      int(dow >= 5),
            "is_peak_morning": int(7 <= hour <= 9),
            "is_peak_evening": int(16 <= hour <= 19),
            "current_speed":         cur_speed,
            "free_flow_speed":       free_speed,
            "current_travel_time":   cur_tt,
            "free_flow_travel_time": free_tt,
            "confidence":   float(fsd.get("confidence", 0.9)),
            "road_closure": int(bool(fsd.get("roadClosure", False))),
            "frc_code":     frc_map.get(fsd.get("frc",""), 4),
            "speed_lag_1":  cur_speed,
            "speed_lag_3":  cur_speed,
            "speed_lag_6":  cur_speed,
            "speed_lag_12": cur_speed,
            "speed_roll_mean_6":  cur_speed,
            "speed_roll_mean_12": cur_speed,
            "speed_roll_std_6":   0.0,
            "tt_ratio_roll_6":    cur_tt / max(free_tt, 1),
            "speed_trend":        0.0,
        }

    col_btn, col_time = st.columns([1, 3])
    with col_btn:
        fetch_live = st.button("Fetch live data", icon=":material/refresh:", type="primary")
    with col_time:
        st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

    if fetch_live:
        if not api_key:
            st.error("Please enter your TomTom API key in the sidebar.")
        else:
            results = []
            progress = st.progress(0, text="Fetching live data...")

            for i, wp_item in enumerate(WAYPOINTS):
                try:
                    fsd   = fetch_tomtom(wp_item["lat"], wp_item["lon"], api_key)
                    feat  = build_features(fsd)
                    X     = np.array([[feat[f] for f in FEATS]])
                    pred  = MODEL.predict(X)[0]
                    proba = MODEL.predict_proba(X)[0]
                    label = LE.inverse_transform([pred])[0]
                    results.append({
                        "name":         wp_item["name"].replace("_"," ").title(),
                        "lat":          wp_item["lat"],
                        "lon":          wp_item["lon"],
                        "label":        label,
                        "confidence":   round(max(proba) * 100, 1),
                        "current_speed": fsd["currentSpeed"],
                        "free_flow_speed": fsd["freeFlowSpeed"],
                        "congested_prob": round(proba[list(LE.classes_).index("congested")] * 100, 1),
                    })
                except Exception as e:
                    st.warning(f"Failed {wp_item['name']}: {e}")
                progress.progress((i+1)/len(WAYPOINTS),
                                  text=f"Fetching {wp_item['name']}...")

            progress.empty()
            st.session_state["live_results"] = results

    results = st.session_state.get("live_results", [])

    # ── Folium heatmap ────────────────────────────────────────────────────
    m = folium.Map(
        location=[48.1372, 11.5755],
        zoom_start=12,
        tiles="CartoDB positron",
    )

    if results:
        for r in results:
            color  = COLORS[r["label"]]
            icon   = {"free_flow":"▲","moderate":"●","congested":"✖"}.get(r["label"],"●")
            radius = {"free_flow": 300, "moderate": 400, "congested": 500}.get(r["label"], 300)

            folium.CircleMarker(
                location=[r["lat"], r["lon"]],
                radius=18,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.75,
                weight=2,
                popup=folium.Popup(
                    f"""
                    <div style='font-family:sans-serif;min-width:160px'>
                    <b style='font-size:14px'>{r['name']}</b><br>
                    <span style='color:{color};font-size:13px'>⬤ {r['label'].replace('_',' ').title()}</span><br>
                    <hr style='margin:4px 0'>
                    <b>Speed:</b> {r['current_speed']} km/h<br>
                    <b>Free flow:</b> {r['free_flow_speed']} km/h<br>
                    <b>Confidence:</b> {r['confidence']}%
                    </div>
                    """,
                    max_width=200,
                ),
                tooltip=f"{r['name']} — {r['label'].replace('_',' ').title()} ({r['confidence']}%)",
            ).add_to(m)

            # Pulse ring for congested
            if r["label"] == "congested":
                folium.CircleMarker(
                    location=[r["lat"], r["lon"]],
                    radius=28,
                    color="#e74c3c",
                    fill=False,
                    weight=1.5,
                    opacity=0.4,
                ).add_to(m)

        # Legend
        legend_html = """
        <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                    background:rgba(255,255,255,0.95);padding:12px 16px;
                    border-radius:8px;border:1px solid #e2e8f0;color:#0f172a;
                    font-family:sans-serif;font-size:13px;
                    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
            <b>Congestion level</b><br>
            <span style="color:#2ecc71">●</span> Free flow<br>
            <span style="color:#f39c12">●</span> Moderate<br>
            <span style="color:#e74c3c">●</span> Congested
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

    st_folium(m, width=None, height=520, returned_objects=[])

    # ── Live results table ────────────────────────────────────────────────
    if results:
        st.markdown("#### Live predictions")
        df_live = pd.DataFrame(results)
        df_live["status"] = df_live["label"].map({
            "free_flow": "Free flow",
            "moderate":  "Moderate",
            "congested": "Congested",
        })
        df_live["speed_display"] = df_live.apply(
            lambda r: f"{r['current_speed']} / {r['free_flow_speed']} km/h", axis=1
        )
        st.dataframe(
            df_live[["name","status","speed_display","confidence"]].rename(columns={
                "name": "Location", "status": "Status",
                "speed_display": "Speed / Free flow", "confidence": "Confidence %"
            }),
            use_container_width=True, hide_index=True,
        )

        # KPI cards
        st.markdown("#### Summary")
        c1, c2, c3, c4 = st.columns(4)
        counts = pd.Series([r["label"] for r in results]).value_counts()
        c1.metric("Free flow",  counts.get("free_flow", 0))
        c2.metric("Moderate",   counts.get("moderate",  0))
        c3.metric("Congested",  counts.get("congested", 0))
        c4.metric("Avg confidence", f"{np.mean([r['confidence'] for r in results]):.1f}%")

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Performance
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Model performance")

    try:
        df = load_processed()
        X  = df[FEATS].values
        y  = LE.transform(df["congestion_level"])
        y_pred  = MODEL.predict(X)
        y_proba = MODEL.predict_proba(X)
        classes = list(LE.classes_)

        col1, col2 = st.columns(2)

        # ── Confusion Matrix ──────────────────────────────────────────────
        with col1:
            st.markdown("#### Confusion matrix")
            cm = confusion_matrix(y, y_pred)
            fig_cm = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=classes, y=classes,
                color_continuous_scale="Blues",
                text_auto=True,
            )
            fig_cm.update_layout(
                height=380,
                margin=dict(l=10,r=10,t=10,b=10),
                font=dict(size=13),
                coloraxis_showscale=False,
            )
            fig_cm.update_traces(textfont_size=16)
            st.plotly_chart(fig_cm, use_container_width=True)

        # ── ROC-AUC ───────────────────────────────────────────────────────
        with col2:
            st.markdown("#### ROC-AUC curve")
            lb  = LabelBinarizer().fit(y)
            y_b = lb.transform(y)
            fig_roc = go.Figure()
            colors_roc = ["#3498db","#2ecc71","#e74c3c"]

            for i, cls in enumerate(classes):
                fpr, tpr, _ = roc_curve(y_b[:, i], y_proba[:, i])
                auc_val = auc(fpr, tpr)
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode="lines",
                    name=f"{cls} (AUC={auc_val:.3f})",
                    line=dict(color=colors_roc[i], width=2.5),
                ))

            fig_roc.add_trace(go.Scatter(
                x=[0,1], y=[0,1], mode="lines",
                line=dict(dash="dash", color="gray", width=1),
                showlegend=False,
            ))
            fig_roc.update_layout(
                height=380,
                xaxis_title="False positive rate",
                yaxis_title="True positive rate",
                legend=dict(x=0.35, y=0.08),
                margin=dict(l=10,r=10,t=10,b=10),
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        # ── Precision-Recall ──────────────────────────────────────────────
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("#### Precision-recall curve")
            fig_pr = go.Figure()
            for i, cls in enumerate(classes):
                prec, rec, _ = precision_recall_curve(y_b[:, i], y_proba[:, i])
                pr_auc = auc(rec, prec)
                fig_pr.add_trace(go.Scatter(
                    x=rec, y=prec, mode="lines",
                    name=f"{cls} (AUC={pr_auc:.3f})",
                    line=dict(color=colors_roc[i], width=2.5),
                ))
            fig_pr.update_layout(
                height=380,
                xaxis_title="Recall",
                yaxis_title="Precision",
                legend=dict(x=0.02, y=0.08),
                margin=dict(l=10,r=10,t=10,b=10),
            )
            st.plotly_chart(fig_pr, use_container_width=True)

        # ── Class distribution ────────────────────────────────────────────
        with col4:
            st.markdown("#### Class distribution")
            dist = df["congestion_level"].value_counts().reset_index()
            dist.columns = ["Class","Count"]
            dist["Color"] = dist["Class"].map(COLORS)
            fig_dist = px.bar(
                dist, x="Class", y="Count",
                color="Class",
                color_discrete_map=COLORS,
                text="Count",
            )
            fig_dist.update_layout(
                height=380,
                showlegend=False,
                margin=dict(l=10,r=10,t=10,b=10),
                xaxis_title="", yaxis_title="Samples",
            )
            fig_dist.update_traces(textposition="outside")
            st.plotly_chart(fig_dist, use_container_width=True)

    except Exception as e:
        st.error(f"Could not load processed data: {e}")

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Feature Analysis
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Feature analysis")

    try:
        df = load_processed()

        col1, col2 = st.columns(2)

        # ── Feature Importance ────────────────────────────────────────────
        with col1:
            st.markdown("#### Feature importance")
            imp = MODEL.feature_importances_
            fi  = pd.DataFrame({"Feature": FEATS, "Importance": imp})
            fi  = fi.sort_values("Importance", ascending=True).tail(15)
            fig_fi = px.bar(
                fi, x="Importance", y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Blues",
            )
            fig_fi.update_layout(
                height=480,
                margin=dict(l=10,r=10,t=10,b=10),
                coloraxis_showscale=False,
                yaxis_title="",
            )
            st.plotly_chart(fig_fi, use_container_width=True)

        # ── Speed distribution by class ───────────────────────────────────
        with col2:
            st.markdown("#### Speed distribution by congestion class")
            fig_box = px.box(
                df, x="congestion_level", y="current_speed",
                color="congestion_level",
                color_discrete_map=COLORS,
                points="outliers",
            )
            fig_box.update_layout(
                height=480,
                showlegend=False,
                margin=dict(l=10,r=10,t=10,b=10),
                xaxis_title="Congestion level",
                yaxis_title="Speed (km/h)",
            )
            st.plotly_chart(fig_box, use_container_width=True)

        # ── Hourly congestion pattern ─────────────────────────────────────
        st.markdown("#### Congestion patterns by hour of day")
        hourly = df.groupby(["hour","congestion_level"]).size().reset_index(name="count")
        fig_hourly = px.bar(
            hourly, x="hour", y="count",
            color="congestion_level",
            color_discrete_map=COLORS,
            barmode="stack",
        )
        fig_hourly.update_layout(
            height=350,
            margin=dict(l=10,r=10,t=10,b=10),
            xaxis_title="Hour of day",
            yaxis_title="Number of readings",
            legend_title="Congestion level",
            xaxis=dict(tickmode="linear", tick0=0, dtick=1),
        )
        st.plotly_chart(fig_hourly, use_container_width=True)

    except Exception as e:
        st.error(f"Could not load data: {e}")

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Historical Trends
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Historical trends")

    try:
        df_raw = load_raw()
        df_loc = df_raw[
            df_raw["location_name"] == wp["name"]
        ].sort_values("timestamp")

        st.markdown(f"#### Speed over time — {selected_location}")
        fig_ts = px.line(
            df_loc, x="timestamp", y="current_speed",
            color_discrete_sequence=["#3498db"],
        )
        fig_ts.add_scatter(
            x=df_loc["timestamp"],
            y=df_loc["free_flow_speed"],
            mode="lines",
            name="Free flow speed",
            line=dict(color="#2ecc71", dash="dash", width=1.5),
        )

        # Shade congested zones
        fig_ts.add_hrect(
            y0=0, y1=df_loc["free_flow_speed"].median() * 0.6,
            fillcolor="#e74c3c", opacity=0.08,
            annotation_text="Congested zone",
            annotation_position="top left",
        )
        fig_ts.update_layout(
            height=380,
            margin=dict(l=10,r=10,t=10,b=10),
            xaxis_title="Time",
            yaxis_title="Speed (km/h)",
            legend=dict(x=0.01, y=0.99),
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Congestion by day of week")
            df_raw["day_name"] = pd.to_datetime(
                df_raw["timestamp"]
            ).dt.day_name()
            day_order = ["Monday","Tuesday","Wednesday",
                         "Thursday","Friday","Saturday","Sunday"]
            day_dist  = df_raw.groupby(
                ["day_name","congestion_level"]
            ).size().reset_index(name="count")
            day_dist["day_name"] = pd.Categorical(
                day_dist["day_name"], categories=day_order, ordered=True
            )
            day_dist = day_dist.sort_values("day_name")
            fig_day = px.bar(
                day_dist, x="day_name", y="count",
                color="congestion_level",
                color_discrete_map=COLORS,
                barmode="stack",
            )
            fig_day.update_layout(
                height=350,
                margin=dict(l=10,r=10,t=10,b=10),
                xaxis_title="",
                yaxis_title="Readings",
                legend_title="",
            )
            st.plotly_chart(fig_day, use_container_width=True)

        with col2:
            st.markdown("#### Weekend vs weekday speed")
            df_raw["period"] = df_raw["is_weekend"].map(
                {0:"Weekday", 1:"Weekend"}
            )
            fig_we = px.violin(
                df_raw, x="period", y="current_speed",
                color="period",
                color_discrete_sequence=["#3498db","#9b59b6"],
                box=True, points=False,
            )
            fig_we.update_layout(
                height=350,
                showlegend=False,
                margin=dict(l=10,r=10,t=10,b=10),
                xaxis_title="",
                yaxis_title="Speed (km/h)",
            )
            st.plotly_chart(fig_we, use_container_width=True)

    except Exception as e:
        st.error(f"Could not load raw data: {e}")

# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — Now vs Usual
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### Now vs usual — is today different?")
    st.caption(
        "Compares current live readings against historical averages "
        "for the same hour and day type (weekday/weekend)."
    )

    try:
        df_raw = load_raw()
        now    = datetime.now(timezone.utc)
        cur_hour    = now.hour
        is_weekend  = int(now.weekday() >= 5)
        period_label = "Weekend" if is_weekend else "Weekday"

        # ── Historical baseline for current hour + day type ───────────────
        df_raw["is_weekend"] = df_raw["is_weekend"].astype(int)
        baseline = (
            df_raw[
                (df_raw["hour"] == cur_hour) &
                (df_raw["is_weekend"] == is_weekend)
            ]
            .groupby("location_name")
            .agg(
                avg_speed   =("current_speed",   "mean"),
                avg_tt      =("current_travel_time", "mean"),
                std_speed   =("current_speed",   "std"),
                sample_count=("current_speed",   "count"),
            )
            .reset_index()
        )

        st.markdown(
            f"#### Baseline: {period_label}s at {cur_hour:02d}:00 "
            f"(from {len(df_raw)} historical readings)"
        )

        live = st.session_state.get("live_results", [])

        if not live:
            st.info(
                "No live data yet — go to **Live Traffic Map** tab "
                "and click **Fetch live data** first, then come back here."
            )
        else:
            # ── Merge live + baseline ─────────────────────────────────────
            df_live = pd.DataFrame(live)
            df_live["location_name"] = df_live["name"].str.lower().str.replace(" ","_")

            merged = df_live.merge(baseline, on="location_name", how="left")
            merged["speed_diff"]   = merged["current_speed"] - merged["avg_speed"]
            merged["pct_change"]   = (merged["speed_diff"] / merged["avg_speed"] * 100).round(1)
            merged["is_slower"]    = merged["speed_diff"] < -3
            merged["is_faster"]    = merged["speed_diff"] > 3

            # ── KPI strip ─────────────────────────────────────────────────
            st.markdown("#### At a glance")
            cols = st.columns(len(merged))
            for i, row in merged.iterrows():
                delta = row["pct_change"]
                arrow = "▼" if delta < 0 else "▲"
                with cols[i]:
                    st.metric(
                        label=row["name"],
                        value=f"{row['current_speed']:.0f} km/h",
                        delta=f"{delta:+.1f}% vs usual",
                        delta_color="inverse",
                    )

            st.markdown("---")

            # ── Side by side bar chart: now vs usual ──────────────────────
            st.markdown("#### Current speed vs historical average (same hour)")
            fig_compare = go.Figure()

            fig_compare.add_trace(go.Bar(
                name="Historical average",
                x=merged["name"],
                y=merged["avg_speed"],
                marker_color="#3498db",
                opacity=0.7,
                error_y=dict(
                    type="data",
                    array=merged["std_speed"].fillna(0),
                    visible=True,
                    color="#2980b9",
                ),
            ))

            fig_compare.add_trace(go.Bar(
                name="Right now",
                x=merged["name"],
                y=merged["current_speed"],
                marker_color=merged["label"].map(COLORS).tolist(),
                opacity=0.9,
            ))

            fig_compare.update_layout(
                barmode="group",
                height=420,
                margin=dict(l=10, r=10, t=10, b=80),
                xaxis_title="",
                yaxis_title="Speed (km/h)",
                legend=dict(x=0.01, y=0.99),
                xaxis_tickangle=-30,
            )
            st.plotly_chart(fig_compare, use_container_width=True)

            # ── Anomaly callouts ──────────────────────────────────────────
            slower = merged[merged["is_slower"]]
            faster = merged[merged["is_faster"]]

            col1, col2 = st.columns(2)
            with col1:
                if len(slower):
                    st.markdown("#### :material/trending_down: Slower than usual")
                    for _, r in slower.iterrows():
                        st.error(
                            f"**{r['name']}** — {r['current_speed']:.0f} km/h now "
                            f"vs {r['avg_speed']:.0f} km/h usual "
                            f"({r['pct_change']:+.1f}%)"
                        )
                else:
                    st.success("No locations significantly slower than usual.")

            with col2:
                if len(faster):
                    st.markdown("#### :material/trending_up: Faster than usual")
                    for _, r in faster.iterrows():
                        st.success(
                            f"**{r['name']}** — {r['current_speed']:.0f} km/h now "
                            f"vs {r['avg_speed']:.0f} km/h usual "
                            f"({r['pct_change']:+.1f}%)"
                        )

            st.markdown("---")

            # ── Weekend vs Weekday pattern for selected location ──────────
            st.markdown(
                f"#### Weekend vs weekday speed pattern — {selected_location}"
            )

            loc_key = wp["name"]
            df_loc  = df_raw[df_raw["location_name"] == loc_key].copy()
            df_loc["period"] = df_loc["is_weekend"].map({0:"Weekday",1:"Weekend"})

            hourly_pattern = (
                df_loc.groupby(["hour","period"])["current_speed"]
                .mean().reset_index()
            )

            fig_pattern = px.line(
                hourly_pattern,
                x="hour", y="current_speed",
                color="period",
                color_discrete_map={"Weekday":"#3498db","Weekend":"#9b59b6"},
                markers=True,
                line_shape="spline",
            )

            # Mark current hour
            fig_pattern.add_vline(
                x=cur_hour,
                line_dash="dash",
                line_color="#64748b",
                opacity=0.5,
                annotation_text=f"Now ({cur_hour:02d}:00)",
                annotation_position="top",
            )

            # Shade rush hours
            fig_pattern.add_vrect(
                x0=7, x1=9,
                fillcolor="#e74c3c", opacity=0.08,
                annotation_text="AM peak",
                annotation_position="top left",
            )
            fig_pattern.add_vrect(
                x0=16, x1=19,
                fillcolor="#e74c3c", opacity=0.08,
                annotation_text="PM peak",
                annotation_position="top left",
            )

            fig_pattern.update_layout(
                height=400,
                margin=dict(l=10, r=10, t=30, b=10),
                xaxis_title="Hour of day",
                yaxis_title="Avg speed (km/h)",
                legend_title="Day type",
                xaxis=dict(tickmode="linear", tick0=0, dtick=1),
            )
            st.plotly_chart(fig_pattern, use_container_width=True)

            # ── Heatmap: speed by hour + day of week ──────────────────────
            st.markdown(
                f"#### Speed heatmap by hour and day — {selected_location}"
            )
            df_loc["day_name"] = pd.to_datetime(
                df_loc["timestamp"]
            ).dt.day_name()
            day_order = ["Monday","Tuesday","Wednesday",
                         "Thursday","Friday","Saturday","Sunday"]

            heat_data = (
                df_loc.groupby(["day_name","hour"])["current_speed"]
                .mean().reset_index()
            )
            heat_pivot = heat_data.pivot(
                index="day_name", columns="hour", values="current_speed"
            ).reindex(day_order)

            fig_heat = px.imshow(
                heat_pivot,
                color_continuous_scale=[
                    [0.0,  "#e74c3c"],
                    [0.5,  "#f39c12"],
                    [1.0,  "#2ecc71"],
                ],
                aspect="auto",
                labels=dict(x="Hour", y="Day", color="Speed km/h"),
            )
            fig_heat.update_layout(
                height=320,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(tickmode="linear", tick0=0, dtick=1),
                coloraxis_colorbar=dict(title="km/h"),
            )
            fig_heat.update_traces(
                hovertemplate="Day: %{y}<br>Hour: %{x}<br>Avg speed: %{z:.1f} km/h"
            )
            st.plotly_chart(fig_heat, use_container_width=True)

            st.caption(
                "Red = slow/congested · Orange = moderate · Green = free flow. "
                "Darker red blocks on weekday mornings/evenings confirm rush hour patterns."
            )

    except Exception as e:
        st.error(f"Error loading comparison data: {e}")
        st.exception(e)
# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — Prediction Accuracy & Drift
# ════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown(f"### {HORIZON_MINS}-Minute Ahead Prediction — Live Accuracy & Drift")
    st.caption(
        f"Predictions are made {HORIZON_MINS} min in advance by `src/evaluate_live.py` "
        "(run every 5 min via cron). Each cycle fetches real TomTom data, "
        "compares it to the stored prediction, and logs the result here."
    )

    @st.cache_data(ttl=60)
    def load_eval_log():
        if not os.path.exists(EVAL_LOG_PATH):
            return pd.DataFrame()
        df = pd.read_csv(EVAL_LOG_PATH, parse_dates=["eval_time"])
        return df.sort_values("eval_time")

    eval_df = load_eval_log()

    if eval_df.empty:
        st.info(
            "⚠️ No evaluation data yet.\n\n"
            "Run `src/evaluate_live.py` every 5 minutes to populate this tab. "
            "After **15 minutes** the first accuracy measurement will appear."
        )
        st.code(
            "# In a terminal (project root, every 5 min):\n"
            "TOMTOM_API_KEY=<your_key> python src/evaluate_live.py",
            language="bash",
        )
    else:
        # ── KPI strip ─────────────────────────────────────────────────────
        last  = eval_df.iloc[-1]
        n_obs = len(eval_df)
        prev_acc = eval_df["accuracy"].iloc[-2] if n_obs > 1 else None

        st.markdown("#### At a glance")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric(
            "Latest accuracy",
            f"{last['accuracy']*100:.1f}%",
            delta=(
                f"{(last['accuracy'] - prev_acc)*100:+.1f}pp"
                if prev_acc is not None else None
            ),
        )
        k2.metric("Latest F1 (weighted)", f"{last['f1_weighted']:.4f}")
        k3.metric("Evaluations logged",   n_obs)
        k4.metric(
            "Speed PSI (data drift)",
            f"{last['speed_psi']:.4f}" if not np.isnan(last["speed_psi"]) else "N/A",
            help="PSI < 0.10: stable | 0.10–0.20: moderate | > 0.20: significant drift",
        )
        k5.metric(
            "Model confidence drift",
            f"{last['model_drift_score']:.4f}" if not np.isnan(last["model_drift_score"]) else "N/A",
            help="1 − avg max-probability. Higher = less confident / more drift.",
        )

        st.markdown("---")

        # ── Accuracy over time ──────────────────────────────────────────────────
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown(f"#### Live accuracy over time ({HORIZON_MINS}-min ahead)")
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(
                x=eval_df["eval_time"], y=eval_df["accuracy"] * 100,
                mode="lines+markers", name="Accuracy %",
                line=dict(color="#3498db", width=2.5),
                marker=dict(size=6),
                fill="tozeroy",
                fillcolor="rgba(52,152,219,0.1)",
            ))
            fig_acc.add_hline(
                y=eval_df["accuracy"].mean() * 100,
                line_dash="dash", line_color="#2ecc71",
                annotation_text=f"Mean {eval_df['accuracy'].mean()*100:.1f}%",
                annotation_position="top right",
            )
            fig_acc.update_layout(
                height=340, margin=dict(l=10, r=10, t=10, b=10),
                yaxis=dict(title="Accuracy %", range=[0, 105]),
                xaxis_title="Time",
            )
            st.plotly_chart(fig_acc, use_container_width=True)

        with col_b:
            st.markdown("#### F1-score (weighted) over time")
            fig_f1 = go.Figure()
            fig_f1.add_trace(go.Scatter(
                x=eval_df["eval_time"], y=eval_df["f1_weighted"],
                mode="lines+markers", name="F1",
                line=dict(color="#9b59b6", width=2.5),
                marker=dict(size=6),
            ))
            fig_f1.update_layout(
                height=340, margin=dict(l=10, r=10, t=10, b=10),
                yaxis=dict(title="Weighted F1", range=[0, 1.05]),
                xaxis_title="Time",
            )
            st.plotly_chart(fig_f1, use_container_width=True)

        # ── Label distribution: actual vs predicted ───────────────────────────
        st.markdown("#### Predicted vs actual label distribution (cumulative)")
        label_cols = ["free_flow", "moderate", "congested"]
        actual_totals = [
            eval_df[f"actual_{c}"].sum() for c in label_cols
        ]
        pred_totals = [
            eval_df[f"pred_{c}"].sum() for c in label_cols
        ]

        fig_labels = go.Figure(data=[
            go.Bar(
                name="Actual",
                x=label_cols, y=actual_totals,
                marker_color=[COLORS[c] for c in label_cols],
                opacity=0.9,
            ),
            go.Bar(
                name="Predicted",
                x=label_cols, y=pred_totals,
                marker_color=[COLORS[c] for c in label_cols],
                opacity=0.45,
                marker_pattern_shape="/",
            ),
        ])
        fig_labels.update_layout(
            barmode="group", height=340,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Congestion class",
            yaxis_title="Total readings",
            legend=dict(x=0.01, y=0.99),
        )
        st.plotly_chart(fig_labels, use_container_width=True)

        # ── Drift signals ─────────────────────────────────────────────────────
        st.markdown("#### Drift monitoring")
        col_d1, col_d2 = st.columns(2)

        with col_d1:
            st.markdown("##### Data drift — Speed PSI over time")
            fig_psi = go.Figure()
            fig_psi.add_trace(go.Scatter(
                x=eval_df["eval_time"],
                y=eval_df["speed_psi"],
                mode="lines+markers", name="PSI",
                line=dict(color="#e74c3c", width=2),
                marker=dict(size=5),
            ))
            # PSI threshold bands
            fig_psi.add_hrect(
                y0=0, y1=0.10, fillcolor="#2ecc71", opacity=0.08,
                annotation_text="Stable (PSI < 0.10)",
                annotation_position="top left",
            )
            fig_psi.add_hrect(
                y0=0.10, y1=0.20, fillcolor="#f39c12", opacity=0.08,
                annotation_text="Moderate drift",
                annotation_position="top left",
            )
            fig_psi.add_hrect(
                y0=0.20, y1=1.0, fillcolor="#e74c3c", opacity=0.08,
                annotation_text="Significant drift",
                annotation_position="top left",
            )
            fig_psi.update_layout(
                height=320, margin=dict(l=10, r=10, t=10, b=10),
                yaxis_title="PSI", xaxis_title="Time",
                yaxis=dict(rangemode="tozero"),
            )
            st.plotly_chart(fig_psi, use_container_width=True)

        with col_d2:
            st.markdown("##### Model drift — Confidence score over time")
            st.caption(
                "Model drift score = 1 − avg max-class probability. "
                "A rising score means the model is becoming less certain "
                "(possible concept drift)."
            )
            fig_drift = go.Figure()
            fig_drift.add_trace(go.Scatter(
                x=eval_df["eval_time"],
                y=eval_df["model_drift_score"],
                mode="lines+markers", name="Drift score",
                line=dict(color="#e67e22", width=2),
                marker=dict(size=5),
                fill="tozeroy",
                fillcolor="rgba(230,126,34,0.08)",
            ))
            # Uniform-random baseline (1/3 classes → score = 1 − 0.333 = 0.667)
            fig_drift.add_hline(
                y=0.667, line_dash="dot", line_color="#e74c3c",
                annotation_text="Random-guess baseline",
                annotation_position="top right",
            )
            fig_drift.update_layout(
                height=320, margin=dict(l=10, r=10, t=10, b=10),
                yaxis_title="Drift score", xaxis_title="Time",
                yaxis=dict(range=[0, 0.8]),
            )
            st.plotly_chart(fig_drift, use_container_width=True)

        # ── Raw log table ─────────────────────────────────────────────────────
        with st.expander("Raw evaluation log"):
            display_df = eval_df.copy()
            display_df["accuracy"] = (display_df["accuracy"] * 100).round(1).astype(str) + "%"
            display_df["f1_weighted"] = display_df["f1_weighted"].round(4)
            st.dataframe(
                display_df[["eval_time", "accuracy", "f1_weighted",
                             "n_correct", "n_total", "speed_psi", "model_drift_score"]]
                .rename(columns={
                    "eval_time":         "Time",
                    "accuracy":          "Accuracy",
                    "f1_weighted":       "F1 (weighted)",
                    "n_correct":         "Correct",
                    "n_total":           "Total",
                    "speed_psi":         "Speed PSI",
                    "model_drift_score": "Drift score",
                })
                .sort_values("Time", ascending=False),
                use_container_width=True, hide_index=True,
            )
