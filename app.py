import gradio as gr
import pandas as pd
import joblib
import shap
import json
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from fpdf import FPDF
from datetime import datetime

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Authentication
CREDENTIALS_FILE = "users.json"
LOG_FILE = "user_logs.xlsx"
DEVELOPER_EMAIL = "arslanhafeezkhan.16@gmail.com"

# Load or initialize users
if not os.path.exists(CREDENTIALS_FILE):
    with open(CREDENTIALS_FILE, "w") as f:
        json.dump({}, f)

with open(CREDENTIALS_FILE, "r") as f:
    users = json.load(f)

current_user = {"email": None}

# Login/Signup
def signup(email, password):
    if email in users:
        return "User already exists."
    users[email] = {"password": password}
    with open(CREDENTIALS_FILE, "w") as f:
        json.dump(users, f)
    current_user["email"] = email
    return "Signup successful."

def login(email, password):
    if users.get(email, {}).get("password") == password:
        current_user["email"] = email
        return "Login successful."
    return "Invalid credentials."

# Predict Function
def predict_strength(data):
    scaled = scaler.transform(data)
    preds = pd.DataFrame({
        "RandomForest": model.predict(scaled),
        "GradientBoost": GradientBoostingRegressor().fit(data, model.predict(scaled)).predict(data),
        "LinearRegression": LinearRegression().fit(data, model.predict(scaled)).predict(data)
    })
    preds["Average"] = preds.mean(axis=1)
    return preds, r2_score(model.predict(scaled), preds["Average"])

# PDF Generator
def generate_pdf(input_data, prediction, shap_fig, contrib_plot):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "AI Concrete Strength Prediction Report", ln=1)

    pdf.set_font("Arial", "", 12)
    for col, val in input_data.items():
        pdf.cell(0, 10, f"{col}: {val}", ln=1)

    pdf.cell(0, 10, f"Predicted Strength (MPa): {prediction:.2f}", ln=1)

    # Save SHAP & Contribution plot
    shap_fig.savefig("shap_plot.png")
    contrib_plot.savefig("contrib_plot.png")

    pdf.image("shap_plot.png", x=10, y=None, w=180)
    pdf.image("contrib_plot.png", x=10, y=None, w=180)

    report_path = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(report_path)
    return report_path

# SHAP Plot
def get_shap_plot(data):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data)
    fig = plt.figure()
    shap.summary_plot(shap_values, data, show=False)
    return fig

# Contribution Plot
def strength_contribution(data_row):
    weights = data_row / data_row.sum()
    fig = plt.figure()
    weights.plot(kind="pie", autopct='%1.1f%%', legend=False, title="Constituent Contribution to Strength")
    return fig

# Save to Excel Log
def log_user_prediction(user, input_data, prediction):
    df = pd.DataFrame([input_data])
    df["Predicted Strength"] = prediction
    df["User"] = user
    df["Time"] = datetime.now()

    if os.path.exists(LOG_FILE):
        existing = pd.read_excel(LOG_FILE)
        df = pd.concat([existing, df], ignore_index=True)

    df.to_excel(LOG_FILE, index=False)

# User History
def view_history():
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame()

    df = pd.read_excel(LOG_FILE)
    if current_user["email"] == DEVELOPER_EMAIL:
        return df
    return df[df["User"] == current_user["email"]]

# Single Prediction UI
def predict_ui(cement, sand, agg, water, sp, fly, slag, age):
    df = pd.DataFrame([[cement, sand, agg, water, sp, fly, slag, age]],
                      columns=["Cement", "Sand", "Coarse Aggregate", "Water", "Superplasticizer", "Fly Ash", "Slag", "Age"])

    preds, r2 = predict_strength(df)
    shap_fig = get_shap_plot(df)
    contrib_fig = strength_contribution(df.iloc[0])
    pdf_path = generate_pdf(df.iloc[0].to_dict(), preds["Average"].iloc[0], shap_fig, contrib_fig)

    log_user_prediction(current_user["email"], df.iloc[0].to_dict(), preds["Average"].iloc[0])
    return preds["Average"].iloc[0], r2, shap_fig, contrib_fig, pdf_path

# Excel Upload UI
def bulk_predict(file):
    df = pd.read_excel(file)
    preds, r2 = predict_strength(df)
    df["Predicted Strength"] = preds["Average"]

    for i, row in df.iterrows():
        log_user_prediction(current_user["email"], row.drop("Predicted Strength").to_dict(), row["Predicted Strength"])

    return df

# Gradio Interface
with gr.Blocks() as demo:
    with gr.Tab("Login/Signup"):
        email = gr.Text(label="Email")
        password = gr.Text(label="Password", type="password")
        login_btn = gr.Button("Login")
        signup_btn = gr.Button("Sign Up")
        auth_output = gr.Textbox(label="Auth Status")

        login_btn.click(fn=login, inputs=[email, password], outputs=auth_output)
        signup_btn.click(fn=signup, inputs=[email, password], outputs=auth_output)

    with gr.Tab("Predict Strength"):
        gr.Markdown("### Enter Mix Design Parameters")
        cement = gr.Number()
        sand = gr.Number()
        agg = gr.Number(label="Coarse Aggregate")
        water = gr.Number()
        sp = gr.Number(label="Superplasticizer")
        fly = gr.Number(label="Fly Ash")
        slag = gr.Number()
        age = gr.Number()

        predict_btn = gr.Button("Predict Strength")
        strength = gr.Textbox(label="Predicted Strength (MPa)")
        r2_score_out = gr.Textbox(label="Model R² Score")
        shap_plot = gr.Plot(label="SHAP Plot")
        contrib_plot = gr.Plot(label="Strength Contribution")
        pdf_download = gr.File(label="Download Report")

        predict_btn.click(fn=predict_ui,
                          inputs=[cement, sand, agg, water, sp, fly, slag, age],
                          outputs=[strength, r2_score_out, shap_plot, contrib_plot, pdf_download])

    with gr.Tab("Bulk Prediction (Excel)"):
        excel_upload = gr.File(label="Upload Excel File", type="filepath")
        excel_output = gr.Dataframe()
        excel_btn = gr.Button("Run Bulk Prediction")

        excel_btn.click(fn=bulk_predict, inputs=[excel_upload], outputs=excel_output)

    with gr.Tab("Prediction History"):
        history_btn = gr.Button("View My History")
        history_output = gr.Dataframe()

        history_btn.click(fn=view_history, outputs=history_output)

demo.launch()
