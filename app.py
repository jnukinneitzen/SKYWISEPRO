import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime

# --- CONFIGURATION ---
API_URL = "http://127.0.0.1:8000/predict"
CHECKWX_API_KEY = "50c133428c79479f9e05d62f33064ead"

st.set_page_config(page_title="SkyWise Pro | Global Aviation AI", layout="wide", page_icon="✈️")

# --- INITIALIZE SESSION STATE ---
if 'multi_data' not in st.session_state:
    st.session_state['multi_data'] = []

# --- MODERN UI STYLING ---
st.markdown("""
    <style>
    /* Force metrics to be visible in dark/light mode */
    [data-testid="stMetricValue"] { color: #0ea5e9 !important; font-size: 1.8rem !important; }
    [data-testid="stMetricLabel"] { color: #64748b !important; font-weight: 600 !important; }
    .stMetric { 
        background-color: #ffffff; 
        padding: 20px; 
        border-radius: 12px; 
        border: 1px solid #e5e7eb; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- DEFENSIVE HELPER: FIXES 'INT' OBJECT HAS NO ATTRIBUTE 'GET' ---
def get_nested_val(data_dict, field, subkey, default):
    """Safely pulls data even if the API returns a raw number instead of a dict."""
    val = data_dict.get(field, default)
    if isinstance(val, dict):
        res = val.get(subkey, default)
        return res if res is not None else default
    # If the API returned a raw int/float directly (like VIDP humidity)
    return val if isinstance(val, (int, float)) else default

# --- MULTI-HUB FETCH LOGIC ---
def fetch_multi_hub(icao_list):
    icao_string = ",".join(icao_list).upper()
    url = f"https://api.checkwx.com/metar/{icao_string}/decoded"
    headers = {'X-API-Key': CHECKWX_API_KEY}
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            st.sidebar.error(f"📡 API Error {response.status_code}. Key may be pending.")
            return

        res = response.json()
        if isinstance(res, dict) and res.get('results', 0) > 0:
            st.session_state['multi_data'] = res.get('data', [])
            st.sidebar.success(f"✅ Loaded {len(st.session_state['multi_data'])} Hubs!")
        else:
            st.sidebar.warning("No data found for selected codes.")
    except Exception as e:
        st.sidebar.error(f"🚨 Connection Error: {str(e)}")

# --- SIDEBAR: FLEET SELECTION ---
st.sidebar.title("✈️ Fleet Control")
hubs = st.sidebar.multiselect(
    "Select Hubs to Monitor:",
    ["KJFK", "KLAX", "KORD", "VIDP", "VABB", "EGLL", "RJTT"],
    default=["KJFK", "VIDP", "KLAX"]
)

if st.sidebar.button("🔄 Refresh Fleet Data", width='stretch'):
    fetch_multi_hub(hubs)

st.sidebar.divider()
st.sidebar.info("System optimized for production-level inference tracking.")

# --- MAIN DASHBOARD ---
st.title(" SKYWISE PRO")
st.markdown("Real-time visibility forecasting using **MoE Architecture** and **SHAP Causal Inference**.")

if st.session_state['multi_data']:
    tab1, tab2 = st.tabs([" Fleet Overview", " Deep-Dive Analysis"])

    # --- TAB 1: FLEET OVERVIEW ---
    with tab1:
        cols = st.columns(len(st.session_state['multi_data']))
        for i, data in enumerate(st.session_state['multi_data']):
            with cols[i]:
                icao = data.get('icao', 'Unknown')
                vis_raw = data.get('visibility', {})
                c_vis = float(vis_raw.get('miles_float', 10.0) if isinstance(vis_raw, dict) else (vis_raw or 10.0))
                
                st.subheader(f" {icao}")
                st.metric("Current Vis", f"{c_vis} mi")
                
                # Defensive Feature Engineering
                dry = round((get_nested_val(data, 'temperature', 'celsius', 20) * 9/5) + 32, 1)
                dew = round((get_nested_val(data, 'dewpoint', 'celsius', 15) * 9/5) + 32, 1)
                
                payload = {
                    "DATE": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "DRYBULBTEMPF": dry, 
                    "WETBULBTEMPF": (dry * 0.352 + dew * 0.648),
                    "DewPointTempF": dew, 
                    "RelativeHumidity": float(get_nested_val(data, 'humidity', 'percent', 50)),
                    "WindSpeed": float(get_nested_val(data, 'wind', 'speed_mph', 5.0)),
                    "WindDirection": float(get_nested_val(data, 'wind', 'degrees', 180)),
                    "StationPressure": float(get_nested_val(data, 'barometer', 'hg', 29.92)),
                    "CURRENT_VISIBILITY": c_vis
                }
                
                # Background Inference for each Column
                try:
                    r = requests.post(API_URL, json=payload)
                    if r.status_code == 200:
                        pred = r.json().get('predicted_visibility', 0)
                        st.metric("AI Forecast", f"{pred} mi", delta=round(pred - c_vis, 2))
                    else:
                        st.caption("AI Syncing...")
                except:
                    st.caption("Backend Offline")

    # --- TAB 2: DEEP-DIVE ANALYSIS ---
    with tab2:
        selected_icao = st.selectbox("Select Hub for Causal Analysis:", [d.get('icao') for d in st.session_state['multi_data']])
        station_data = next(d for d in st.session_state['multi_data'] if d.get('icao') == selected_icao)
        
        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.write(f"### {selected_icao} Stats")
            # Extract values for display
            s_dry = round((get_nested_val(station_data, 'temperature', 'celsius', 20) * 9/5) + 32, 1)
            s_hum = get_nested_val(station_data, 'humidity', 'percent', 50)
            
            st.info(f"**Temperature:** {s_dry}°F")
            st.info(f"**Humidity:** {s_hum}%")
            st.info(f"**Wind:** {get_nested_val(station_data, 'wind', 'speed_mph', 0)} mph")

        with c2:
            if st.button(f"🔍 Run SHAP Analysis for {selected_icao}", width='stretch'):
                # Construct payload for SHAP request
                s_dew = round((get_nested_val(station_data, 'dewpoint', 'celsius', 15) * 9/5) + 32, 1)
                s_vis_raw = station_data.get('visibility', {})
                s_c_vis = float(s_vis_raw.get('miles_float', 10.0) if isinstance(s_vis_raw, dict) else (s_vis_raw or 10.0))
                
                s_payload = {
                    "DATE": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "DRYBULBTEMPF": s_dry, "WETBULBTEMPF": (s_dry * 0.352 + s_dew * 0.648),
                    "DewPointTempF": s_dew, "RelativeHumidity": float(s_hum),
                    "WindSpeed": float(get_nested_val(station_data, 'wind', 'speed_mph', 5.0)),
                    "WindDirection": float(get_nested_val(station_data, 'wind', 'degrees', 180)),
                    "StationPressure": float(get_nested_val(station_data, 'barometer', 'hg', 29.92)),
                    "CURRENT_VISIBILITY": s_c_vis
                }
                
                try:
                    res_r = requests.post(API_URL, json=s_payload)
                    if res_r.status_code == 200:
                        res_data = res_r.json()
                        shap_values = res_data.get('shap_values', {})
                        if shap_values:
                            shap_df = pd.DataFrame(shap_values.items(), columns=['Feature', 'Impact']).sort_values('Impact')
                            fig = px.bar(shap_df, x='Impact', y='Feature', orientation='h', color='Impact', color_continuous_scale='RdBu_r')
                            st.plotly_chart(fig, width='stretch')
                except Exception as e:
                    st.error(f"Inference Error: {e}")

else:
    st.info(f" Welcome, Select your primary hubs in the sidebar to begin fleet monitoring.")

st.divider()
st.caption("Nikunj Bisht | NSUT ECE 2023UEC4585 | Production Interpretable System")
