import gradio as gr
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import openpyxl
from datetime import datetime
from fpdf import FPDF
import os

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Email of developer
DEVELOPER_EMAIL = "arslanhafeezkhan.16@gmail.com"
LOG_FILE = "user_logs.xlsx"

# Prediction function
def predict_strength(inputs):
    X = scaler.transform([inputs])
    preds = model.predict(X)
    return np.round(np.mean(preds), 2), preds

# Save user log
def save_log(email, inputs, strength):
    df = pd.DataFrame([[datetime.now(), email] + inputs + [strength]])
    df.to_excel(LOG_FILE, mode='a', header=False, index=False)

# View history
def view_history(email):
    if not os.path.exists(LOG_FILE):
        return "No logs yet."
    df = pd.read_excel(LOG_FILE)
    if email != DEVELOPER_EMAIL:
        df = df[df['User Email'] == email]
    return df.tail(10)

# Generate SHAP summary plot
def explain_prediction(inputs):
    explainer = shap.Explainer(model, masker=scaler.transform)
    shap_values = explainer([inputs])
    fig, ax = plt.subplots()
    shap.plots.bar(shap_values[0], show=False)
    plt.title("SHAP Feature Importance")
    fig.savefig("shap_plot.png")
    plt.close(fig)
    return "shap_plot.png"

# Generate constituent strength pie chart
def constituent_chart(inputs):
    weights = np.array(inputs) / np.sum(inputs)
    labels = ['Cement', 'Sand', 'Coarse Agg.', 'Water', 'Superpl.', 'Fly Ash', 'Slag', 'Age']
    fig, ax = plt.subplots()
    ax.pie(weights, labels=labels, autopct='%1.1f%%')
    plt.title("Strength Contribution (%)")
    fig.savefig("pie_chart.png")
    plt.close(fig)
    return "pie_chart.png"

# Generate PDF Report
def generate_pdf(email, inputs, strength, methods, reason_img, contrib_img):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="AI Concrete Strength Prediction Report", ln=1, align="C")
    pdf.cell(200, 10, txt=f"User: {email}", ln=1)

    labels = ['Cement', 'Sand', 'Coarse Agg.', 'Water', 'Superpl.', 'Fly Ash', 'Slag', 'Age']
    for i, val in enumerate(inputs):
        pdf.cell(200, 10, txt=f"{labels[i]}: {val}", ln=1)

    pdf.cell(200, 10, txt=f"\nFinal Predicted Strength: {strength} MPa", ln=1)
    pdf.cell(200, 10, txt=f"Method-wise Predictions: {', '.join([str(round(p,2)) for p in methods])}", ln=1)

    pdf.cell(200, 10, txt="\nReasoning based on input influence:", ln=1)
    pdf.image(reason_img, x=10, y=None, w=180)

    pdf.cell(200, 10, txt="\nConstituent Contribution Chart:", ln=1)
    pdf.image(contrib_img, x=10, y=None, w=180)

    pdf.cell(200, 10, txt="\nSuggestions:", ln=1)
    pdf.multi_cell(0, 10, txt=(
        "- Cement and Superplasticizer improve strength.\n"
        "- Fly Ash and Slag help long-term durability.\n"
        "- Control water for better performance.\n"
        "- Higher curing age generally leads to better strength."
    ))

    pdf.output("report.pdf")
    return "report.pdf"

# Upload Excel
def handle_excel(file, email):
    df = pd.read_excel(file)
    required_cols = ['Cement', 'Sand', 'Coarse Aggregate', 'Water', 'Superplasticizer',
                     'Fly Ash', 'Slag', 'Age']
    if not all(col in df.columns for col in required_cols):
        return "Invalid Excel format"
    inputs = df[required_cols].values
    results = []
    for row in inputs:
        strength, _ = predict_strength(list(row))
        save_log(email, list(row), strength)
        results.append(strength)
    df['Predicted Strength (MPa)'] = results
    df.to_excel("bulk_results.xlsx", index=False)
    return "bulk_results.xlsx"

# Main interface
def main(email, cement, sand, agg, water, superp, flyash, slag, age):
    inputs = [cement, sand, agg, water, superp, flyash, slag, age]
    strength, methods = predict_strength(inputs)
    save_log(email, inputs, strength)
    reason_img = explain_prediction(inputs)
    contrib_img = constituent_chart(inputs)
    pdf_path = generate_pdf(email, inputs, strength, methods, reason_img, contrib_img)
    return strength, reason_img, contrib_img, pdf_path

# Gradio UI
with gr.Blocks(title="AI Concrete Strength Predictor") as app:
    with gr.Tab("Predict"):
        email = gr.Textbox(label="Email")
        cement = gr.Number(label="Cement (kg)")
        sand = gr.Number(label="Sand (kg)")
        agg = gr.Number(label="Coarse Aggregate (kg)")
        water = gr.Number(label="Water (kg)")
        superp = gr.Number(label="Superplasticizer (kg)")
        flyash = gr.Number(label="Fly Ash (kg)")
        slag = gr.Number(label="Slag (kg)")
        age = gr.Number(label="Age (days)")
        btn = gr.Button("Predict Strength")
        result = gr.Number(label="Predicted Strength (MPa)")
        shap_img = gr.Image()
        pie_img = gr.Image()
        pdf_out = gr.File(label="Download PDF Report")
        btn.click(main, inputs=[email, cement, sand, agg, water, superp, flyash, slag, age],
                  outputs=[result, shap_img, pie_img, pdf_out])

    with gr.Tab("History"):
        email2 = gr.Textbox(label="Enter Email to View History")
        btn2 = gr.Button("Show History")
        out_table = gr.Dataframe()
        btn2.click(view_history, inputs=[email2], outputs=out_table)

    with gr.Tab("Bulk Upload"):
        email3 = gr.Textbox(label="Email")
        file = gr.File(label="Upload Excel File", file_types=[".xlsx"])
        btn3 = gr.Button("Submit File")
        output_file = gr.File(label="Download Results")
        btn3.click(handle_excel, inputs=[file, email3], outputs=output_file)

app.launch()
