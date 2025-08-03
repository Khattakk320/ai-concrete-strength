import gradio as gr
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
import os
from fpdf import FPDF
import uuid

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
log_file = "user_logs.xlsx"

# Feature names
features = ["Cement", "Sand", "Coarse Aggregate", "Water", "Superplasticizer", "Fly Ash", "Slag", "Age"]

def predict_strength(email, cement, sand, agg, water, sp, flyash, slag, age):
    inputs = [cement, sand, agg, water, sp, flyash, slag, age]
    X = scaler.transform([inputs])
    prediction = model.predict(X)[0]
    prediction = round(prediction, 2)

    # Contribution chart (dummy logic for now – replace with SHAP or similar)
    total = sum(inputs)
    contribs = [round((x / total) * prediction, 2) for x in inputs]

    # Save to Excel log
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_row = [timestamp, email, *inputs, prediction]

    if os.path.exists(log_file):
        df = pd.read_excel(log_file)
        df.loc[len(df)] = new_row
    else:
        df = pd.DataFrame([new_row], columns=[
            "Timestamp", "User Email", *features, "Predicted Strength (MPa)"
        ])
    df.to_excel(log_file, index=False)

    # Graph 1: Contribution bar chart
    plt.figure(figsize=(8, 4))
    plt.bar(features, contribs, color="skyblue")
    plt.xticks(rotation=45)
    plt.ylabel("Strength Contribution (MPa)")
    plt.title("Constituent-wise Strength Contribution")
    contrib_chart = f"contrib_{uuid.uuid4()}.png"
    plt.tight_layout()
    plt.savefig(contrib_chart)
    plt.close()

    # Graph 2: Pie chart of percent contributions
    percentages = [round(x / prediction * 100, 2) for x in contribs]
    plt.figure(figsize=(6, 6))
    plt.pie(percentages, labels=features, autopct="%1.1f%%")
    plt.title("Constituent Contribution (%)")
    pie_chart = f"pie_{uuid.uuid4()}.png"
    plt.savefig(pie_chart)
    plt.close()

    # PDF Report
    pdf_path = f"report_{uuid.uuid4()}.pdf"
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Concrete Strength Prediction Report", ln=True, align="C")

    pdf.set_font("Arial", "", 12)
    pdf.ln(10)
    pdf.cell(0, 10, f"Email: {email}", ln=True)
    pdf.cell(0, 10, f"Predicted Strength: {prediction} MPa", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Inputs:", ln=True)
    pdf.set_font("Arial", "", 12)
    for f, v in zip(features, inputs):
        pdf.cell(0, 10, f"{f}: {v}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Suggestions & Analysis:", ln=True)
    pdf.set_font("Arial", "", 11)
    reasons = [
        "Cement and water ratio majorly affect strength",
        "Too much sand may reduce durability",
        "Proper curing (Age) helps gain strength",
        "Supplementary materials like Fly Ash and Slag improve later-age strength"
    ]
    for r in reasons:
        pdf.multi_cell(0, 10, f"- {r}")

    # Add contribution chart
    pdf.image(contrib_chart, x=10, y=None, w=180)
    os.remove(contrib_chart)
    os.remove(pie_chart)
    pdf.output(pdf_path)

    return prediction, pie_chart, pdf_path

def view_logs(email):
    if not os.path.exists(log_file):
        return pd.DataFrame()

    df = pd.read_excel(log_file)
    if email == "arslanhafeezkhan.16@gmail.com":
        return df
    return df[df["User Email"] == email]

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 AI Concrete Strength Predictor")

    with gr.Row():
        email = gr.Textbox(label="Email", placeholder="Enter your email to log results")

    with gr.Row():
        cement = gr.Number(label="Cement (kg/m³)")
        sand = gr.Number(label="Sand (kg/m³)")
        agg = gr.Number(label="Coarse Aggregate (kg/m³)")
        water = gr.Number(label="Water (kg/m³)")

    with gr.Row():
        sp = gr.Number(label="Superplasticizer (kg/m³)")
        flyash = gr.Number(label="Fly Ash (kg/m³)")
        slag = gr.Number(label="Slag (kg/m³)")
        age = gr.Number(label="Age (days)")

    predict_btn = gr.Button("Predict Strength")
    output_strength = gr.Textbox(label="Predicted Strength (MPa)")
    output_chart = gr.Image(label="Constituent % Contribution")
    output_pdf = gr.File(label="Download PDF Report")

    predict_btn.click(
        fn=predict_strength,
        inputs=[email, cement, sand, agg, water, sp, flyash, slag, age],
        outputs=[output_strength, output_chart, output_pdf]
    )

    gr.Markdown("## 📜 View Your Prediction History")
    view_btn = gr.Button("Show My Logs")
    log_table = gr.Dataframe(label="Your Logs", interactive=False)
    view_btn.click(fn=view_logs, inputs=[email], outputs=log_table)

demo.launch()
