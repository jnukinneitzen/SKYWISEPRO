import os
import joblib
import numpy as np
import pandas as pd
import shap  
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Import your custom DB logic
try:
    from database import save_prediction 
except ImportError:
    def save_prediction(data, cluster, pred): return "dev_mode_no_id"

app = FastAPI(title="SkyWise Weather Expert System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

last_visibility_observed = None

# --- LOAD MODELS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

try:
    kmeans_model = joblib.load(os.path.join(MODELS_DIR, "final_kmeans.pkl"))
    scalers = {i: joblib.load(os.path.join(MODELS_DIR, f"scaler_cluster_{i}.pkl")) for i in range(3)}
    experts = {i: joblib.load(os.path.join(MODELS_DIR, f"rf_expert_cluster_{i}.pkl")) for i in range(3)}
    
    # Pre-initialize SHAP explainers for each expert to save time during requests
    explainers = {i: shap.TreeExplainer(experts[i]) for i in range(3)}
    
    print("✅ System Ready: K-Means + 3 RF Experts + SHAP Explainers")
except Exception as e:
    print(f"❌ Initialization Error: {e}")

class WeatherInput(BaseModel):
    DATE: str 
    DRYBULBTEMPF: float
    WETBULBTEMPF: float
    DewPointTempF: float
    RelativeHumidity: float
    WindSpeed: float
    WindDirection: float
    StationPressure: float
    CURRENT_VISIBILITY: float 

@app.post("/predict")
async def predict_visibility(data: WeatherInput):
    global last_visibility_observed
    try:
        # 1. LAG & TREND
        if last_visibility_observed is None:
            calc_lag = data.CURRENT_VISIBILITY
            calc_trend = 0.0
        else:
            calc_lag = last_visibility_observed
            calc_trend = data.CURRENT_VISIBILITY - last_visibility_observed
        
        last_visibility_observed = data.CURRENT_VISIBILITY

        # 2. FEATURE ENGINEERING
        dt = datetime.strptime(data.DATE, "%Y-%m-%d %H:%M:%S")
        month_sin, month_cos = np.sin(2 * np.pi * dt.month / 12), np.cos(2 * np.pi * dt.month / 12)
        hour_sin, hour_cos = np.sin(2 * np.pi * dt.hour / 24), np.cos(2 * np.pi * dt.hour / 24)
        dp_dep, wb_dep = data.DRYBULBTEMPF - data.DewPointTempF, data.DRYBULBTEMPF - data.WETBULBTEMPF

        # 3. K-MEANS (10 FEATURES)
        km_input = pd.DataFrame([{
            'DRYBULBTEMPF': data.DRYBULBTEMPF, 'RelativeHumidity': data.RelativeHumidity,
            'WindSpeed': data.WindSpeed, 'StationPressure': data.StationPressure,
            'month_sin': month_sin, 'month_cos': month_cos,
            'hour_sin': hour_sin, 'hour_cos': hour_cos,
            'DewPointDepression': dp_dep, 'WetBulbDepression': wb_dep
        }])
        cluster_id = int(kmeans_model.predict(km_input.values)[0])

        # 4. RANDOM FOREST (13 FEATURES)
        rf_input = pd.DataFrame([{
            'DRYBULBTEMPF': data.DRYBULBTEMPF, 'RelativeHumidity': data.RelativeHumidity,
            'WindSpeed': data.WindSpeed, 'WindDirection': data.WindDirection,
            'StationPressure': data.StationPressure, 'month_sin': month_sin,
            'month_cos': month_cos, 'DewPointDepression': dp_dep,
            'WetBulbDepression': wb_dep, 'hour_sin': hour_sin,
            'hour_cos': hour_cos, 'Visibility_Lag1': calc_lag,
            'Visibility_Trend': calc_trend
        }])
        
        rf_scaled = scalers[cluster_id].transform(rf_input)
        # Convert back to DF so SHAP has feature names
        rf_scaled_df = pd.DataFrame(rf_scaled, columns=rf_input.columns)
        prediction = float(experts[cluster_id].predict(rf_scaled_df)[0])

        # --- 5. SHAP CAUSAL INFERENCE ---
        shap_values = explainers[cluster_id].shap_values(rf_scaled_df)
        # For RF, shap_values might be a list (for classification) or array (regression)
        # Assuming regression:
        shap_contributions = dict(zip(rf_input.columns, shap_values[0].tolist()))

        # 6. DB LOGGING
        save_payload = data.model_dump()
        save_payload.update({"Visibility_Lag1": calc_lag, "Visibility_Trend": calc_trend})
        mongo_id = save_prediction(save_payload, cluster_id, prediction)

        return {
            "predicted_visibility": round(prediction, 4),
            "cluster_id": cluster_id,
            "shap_values": shap_contributions, # <--- Frontend will use this!
            "calculated_metrics": {"lag": calc_lag, "trend": calc_trend, "dp_depression": dp_dep},
            "db_reference": str(mongo_id)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference Error: {str(e)}")

@app.get("/")
async def status():
    return {"service": "SkyWise Pro API", "online": True}