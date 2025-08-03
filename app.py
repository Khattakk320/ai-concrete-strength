import gradio as gr
import json
import os
import pickle
from fpdf import FPDF
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

# Load model and scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load or initialize user credentials
USER_FILE = "users.json"
LOG_FILE = "user_logs.xlsx"

if not os.path.exists(USER_FILE):
    with open(USER_FILE, "w") as f:
        json.dump({}, f)

if not os.path.exists(LOG_FILE):
    df = pd.DataFrame(columns=["Email", "Inputs", "Prediction", "Timestamp"])
    df.to_excel(LOG_FILE, index=False)

# Basic CAPTCHA generator
import random
def generate_captcha():
    return str(random.randint(1000, 9999))

active_sessions = {}

def signup(email, password, captcha_input, actual_captcha, remember_me):
    with open(USER_FILE, "r") as f:
        users = json.load(f)
    if email in users:
        return "❌ Email already registered", None
    if captcha_input != actual_captcha:
        return "❌ CAPTCHA incorrect", None
    users[email] = password
    with open(USER_FILE, "w") as f:
        json.dump(users, f)
    active_sessions[email] = remember_me
    return "✅ Signup successful", email

def login(email, password, captcha_input, actual_captcha, remember_me):
    with open(USER_FILE, "r") as f:
        users = json.load(f)
    if users.get(email) != password:
        return "❌ Invalid credentials", None
    if captcha_input != actual_captcha:
        return "❌ CAPTCHA incorrect", None
    active_sessions[email] = remember_me
    return "✅ Login successful", email

def predict_strength(email, cement, sand, agg, water, superp, flyash, slag, age):
    inputs = [[cement, sand, agg, water, superp, flyash, slag, age]]
    scaled_inputs = scaler.transform(inputs)
    prediction = model.predict(scaled_inputs)[0]
    result = round(prediction, 2)

    log_entry = {
        "Email": email,
        "Inputs": str(inputs),
        "Prediction": result,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    df = pd.read_excel(LOG_FILE)
    df = df.append(log_entry, ignore_index=True)
    df.to_excel(LOG_FILE, index=False)

    return f"Predicted Strength: {result} MPa"

def upload_excel(email, file):
    df = pd.read_excel(file.name)
    predictions = []
    for _, row in df.iterrows():
        input_data = row.values.tolist()
        scaled = scaler.transform([input_data])
        pred = model.predict(scaled)[0]
        predictions.append(round(pred, 2))
    df["Predicted Strength"] = predictions
    return df

def generate_pdf_report(email):
    df = pd.read_excel(LOG_FILE)
    user_df = df[df["Email"] == email]

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Concrete Strength Prediction Report", ln=True, align='C')
    pdf.ln(10)
    for index, row in user_df.iterrows():
        line = f"{row['Timestamp']}: {row['Inputs']} → {row['Prediction']} MPa"
        pdf.multi_cell(0, 10, line)
    report_name = f"{email}_report.pdf"
    pdf.output(report_name)
    return report_name

def plot_results(email):
    df = pd.read_excel(LOG_FILE)
    df = df[df["Email"] == email]
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp")

    plt.figure(figsize=(6, 3))
    plt.plot(df["Timestamp"], df["Prediction"], marker='o')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.title("Predicted Strength Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("MPa")
    plot_file = f"{email}_plot.png"
    plt.savefig(plot_file)
    return plot_file

# Gradio UI

with gr.Blocks() as app:
    with gr.Tabs():
        with gr.Tab("Login / Signup"):
            with gr.Row():
                email = gr.Textbox(label="Email")
                password = gr.Textbox(label="Password", type="password")
            with gr.Row():
                captcha = gr.Textbox(label="Enter CAPTCHA")
                captcha_display = gr.Textbox(value=generate_captcha(), label="CAPTCHA", interactive=False)
                remember_me = gr.Checkbox(label="Remember Me")
            login_btn = gr.Button("Login")
            signup_btn = gr.Button("Sign Up")
            status = gr.Textbox(label="Status")
            current_user = gr.State("")

            login_btn.click(login, [email, password, captcha, captcha_display, remember_me], [status, current_user])
            signup_btn.click(signup, [email, password, captcha, captcha_display, remember_me], [status, current_user])

        with gr.Tab("Predict"):
            with gr.Column():
                cement = gr.Number(label="Cement")
                sand = gr.Number(label="Sand")
                agg = gr.Number(label="Coarse Aggregate")
                water = gr.Number(label="Water")
                superp = gr.Number(label="Superplasticizer")
                flyash = gr.Number(label="Fly Ash")
                slag = gr.Number(label="Slag")
                age = gr.Number(label="Age (days)")
                predict_btn = gr.Button("Predict")
                result = gr.Textbox(label="Result")

                predict_btn.click(predict_strength, [current_user, cement, sand, agg, water, superp, flyash, slag, age], result)

        with gr.Tab("Excel Upload"):
            file_input = gr.File(label="Upload Excel File")
            table_output = gr.Dataframe()
            upload_btn = gr.Button("Upload & Predict")
            upload_btn.click(upload_excel, [current_user, file_input], table_output)

        with gr.Tab("Report & Plot"):
            report_btn = gr.Button("Download PDF Report")
            plot_btn = gr.Button("Show Strength Plot")
            pdf_output = gr.File()
            plot_output = gr.Image()
            report_btn.click(generate_pdf_report, current_user, pdf_output)
            plot_btn.click(plot_results, current_user, plot_output)

app.launch()
