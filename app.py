import gradio as gr
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
from fpdf import FPDF
import os
from datetime import datetime
import openpyxl

# Load model and scaler
model = joblib.load("model.pk1")
scaler = joblib.load("scaler.pk1")

DEVELOPER_EMAIL = "arslanhafeezkhan.16@gmail.com"
LOG_FILE = "user_logs.xlsx"
features = ["Cement", "Sand", "Coarse Aggregate", "Water", "Superplasticizer", "Fly Ash", "Slag", "Age"]

def log_prediction(email, input_data, prediction):
    data = input_data.copy()
    data['Email'] = email
    data['Prediction'] = prediction
    data['Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame([data])
    if not os.path.exists(LOG_FILE):
        df.to_excel(LOG_FILE, index=False)
    else:
        old_df = pd.read_excel(LOG_FILE)
        pd.concat([old_df, df], ignore_index=True).to_excel(LOG_FILE, index=False)

def view_history(email):
    if os.path.exists(LOG_FILE):
        df = pd.read_excel(LOG_FILE)
        return df if email == DEVELOPER_EMAIL else df[df['Email'] == email]
    return pd.DataFrame()

def generate_pdf(input_data, prediction, importances, suggestion_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Concrete Strength Prediction Report", ln=True, align="C")
    pdf.ln(10)

    for key, value in input_data.items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)

    pdf.ln(5)
    pdf.cell(200, 10, txt=f"Predicted Strength: {prediction:.2f} MPa", ln=True)
    pdf.ln(10)

    pdf.cell(200, 10, txt="Feature Importances (%):", ln=True)
    for name, val in importances.items():
        pdf.cell(200, 10, txt=f"{name}: {val:.2f}%", ln=True)

    pdf.ln(10)
    pdf.multi_cell(200, 10, txt="Suggestions:\n" + suggestion_text)

    file_name = f"Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(file_name)
    return file_name

def get_suggestions(importances):
    sorted_features = sorted(importances.items(), key=lambda x: -x[1])
    suggestions = ""
    for name, _ in sorted_features[:3]:
        suggestions += f"- {name} significantly affects strength. Adjust its proportion carefully.\n"
    return suggestions

def predict_strength(email, cement, sand, coarse_agg, water, superplasticizer, fly_ash, slag, age):
    inputs = {
        "Cement": cement,
        "Sand": sand,
        "Coarse Aggregate": coarse_agg,
        "Water": water,
        "Superplasticizer": superplasticizer,
        "Fly Ash": fly_ash,
        "Slag": slag,
        "Age": age
    }
    df = pd.DataFrame([inputs])
    scaled = scaler.transform(df)
    prediction = model.predict(scaled)[0]

    log_prediction(email, inputs, prediction)

    try:
        importances_raw = model.feature_importances_
        total = np.sum(importances_raw)
        importances = {features[i]: 100 * importances_raw[i] / total for i in range(len(features))}
    except:
        importances = {name: 0 for name in features}

    fig = go.Figure(data=[go.Pie(labels=list(importances.keys()), values=list(importances.values()), hole=0.3)])
    fig.update_layout(title="Feature Contribution")
    chart_path = "importance_chart.png"
    fig.write_image(chart_path)

    suggestion_text = get_suggestions(importances)
    report_path = generate_pdf(inputs, prediction, importances, suggestion_text)

    return prediction, importances, chart_path, report_path

def bulk_predict(email, file):
    df = pd.read_excel(file.name)
    results = []
    for i, row in df.iterrows():
        pred, *_ = predict_strength(email, *row.values)
        row_result = row.to_dict()
        row_result["Prediction"] = pred
        results.append(row_result)
    return pd.DataFrame(results)

with gr.Blocks() as app:
    gr.Markdown("# AI Concrete Strength Predictor")

    email = gr.Textbox(label="Your Email")

    with gr.Tab("Manual Prediction"):
        inputs = [gr.Number(label=label) for label in features]
        predict_btn = gr.Button("Predict")
        result = gr.Textbox(label="Predicted Strength (MPa)")
        import_txt = gr.JSON(label="Contribution %")
        import_img = gr.Image(label="Feature Chart")
        pdf_out = gr.File(label="Download Report")

    with gr.Tab("Bulk Prediction"):
        file_input = gr.File(label="Upload Excel (.xlsx)")
        bulk_out = gr.Dataframe()
        bulk_btn = gr.Button("Predict Bulk")

    with gr.Tab("View History"):
        view_btn = gr.Button("View My History")
        history_table = gr.Dataframe()

    predict_btn.click(fn=predict_strength, inputs=[email] + inputs, outputs=[result, import_txt, import_img, pdf_out])
    bulk_btn.click(fn=bulk_predict, inputs=[email, file_input], outputs=bulk_out)
    view_btn.click(fn=view_history, inputs=[email], outputs=history_table)

app.launch()
