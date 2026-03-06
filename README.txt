SkyWise Pro: Interpretable Aviation AI

Automated Mixture of Experts (MoE) Visibility Forecasting System
SkyWise-Pro is a high-fidelity, real-time weather forecasting dashboard designed for global aviation hubs. It leverages a sophisticated Mixture of Experts architecture, using K-Means clustering to route atmospheric profiles to specialized Random Forest models.

The system provides not just predictions, but causal explanations via SHAP, making it a "Glass-Box" AI solution suitable for high-stakes aviation decision-making.


Core Features
Multi-Hub Fleet Monitoring: Real-time METAR data integration for multiple international airports (KJFK, VIDP, KLAX, etc.) via the CheckWX API.

Intelligent Routing: Uses K-Means to categorize current atmospheric conditions (e.g., high-humidity fog vs. high-pressure clearings) and assigns them to the most accurate "Expert" model.

Explainable AI (XAI): Integrated SHAP (SHapley Additive exPlanations) charts to visualize exactly how features like Relative Humidity or Station Pressure impacted the specific visibility forecast.

Production-Grade Reliability: Implements defensive parsing to handle inconsistent API responses (e.g., handling both dictionary and scalar data types from sensors).

Enterprise UI/UX: A streamlined Streamlit interface featuring a Fleet Overview for dispatchers and a Deep-Dive Analysis tab for meteorologists.