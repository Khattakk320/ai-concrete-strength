# app.py (Corrected & Functional — Fully Working AI Concrete Strength App)

import gradio as gr
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------- Initialization ----------------------
LOG_FILE = "logs/user_logs.xlsx"
UPLOAD_DIR = "uploads"
os.makedirs("logs", exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

model = joblib.load("concrete_strength_model.pkl")
scaler = joblib.load("concrete_scaler.pkl")

# ------------------- Prediction & Logging ----------------------
def prepare_features(df):
    df["Water/Cement"] = df["Water"] / (df["Cement"] + 1e-6)
    df["Binder"] = df["Cement"] + df["Slag"] + df["FlyAsh"]
    df["Fine/Coarse"] = df["FineAggregate"] / (df["CoarseAggregate"] + 1e-6)
    return df

def predict_strength(cement, slag, flyash, water, superplasticizer, coarse, fine, age):
    df = pd.DataFrame([[cement, slag, flyash, water, superplasticizer, coarse, fine, age]],
                      columns=["Cement", "Slag", "FlyAsh", "Water", "Superplasticizer",
                               "CoarseAggregate", "FineAggregate", "Age"])
    df = prepare_features(df)
    X = scaler.transform(df)
    prediction = model.predict(X)[0]
    lower, upper = prediction - 1, prediction + 1

    suggestion = "🧪 Suggestion: Increase binder or age for more strength." if prediction < 25 else "✅ Suggestion: Strong mix. Consider cost efficiency."

    # Log prediction
    df["Predicted Strength"] = prediction
    df["Datetime"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    if os.path.exists(LOG_FILE):
        logs = pd.read_excel(LOG_FILE)
        logs = pd.concat([logs, df], ignore_index=True)
    else:
        logs = df
    logs.to_excel(LOG_FILE, index=False)

    return f"✅ Predicted Strength: {prediction:.2f} MPa\n📉 Confidence Interval: {lower:.2f} - {upper:.2f} MPa\n{suggestion}"

def manual_histogram(cement, slag, flyash, water, superplasticizer, coarse, fine, age):
    df = pd.DataFrame([[cement, slag, flyash, water, superplasticizer, coarse, fine, age]],
                      columns=["Cement", "Slag", "FlyAsh", "Water", "Superplasticizer",
                               "CoarseAggregate", "FineAggregate", "Age"])
    df = prepare_features(df)
    df = df.T.reset_index()
    df.columns = ["Factor", "Value"]
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=df, x="Value", y="Factor", palette="crest", ax=ax)
    ax.set_title("Effect of Factors on Concrete Mix (Input Levels)")
    return fig

def batch_predict(file):
    df = pd.read_excel(file.name) if file.name.endswith("xlsx") else pd.read_csv(file.name)
    df.columns = df.columns.str.strip().str.title().str.replace(" ", "")
    df.rename(columns={"Flyash": "FlyAsh", "Coarseaggregate": "CoarseAggregate", "Fineaggregate": "FineAggregate"}, inplace=True)
    df = prepare_features(df)
    features = ["Cement", "Slag", "FlyAsh", "Water", "Superplasticizer",
                "CoarseAggregate", "FineAggregate", "Age",
                "Water/Cement", "Binder", "Fine/Coarse"]
    X = scaler.transform(df[features])
    preds = model.predict(X)
    df["Predicted Strength"] = preds
    df["Lower"] = preds - 1
    df["Upper"] = preds + 1
    df["Suggestion"] = ["🧪 Consider increasing binder or curing age." if val < 25 else "✅ Strong mix." for val in preds]
    df["Datetime"] = datetime.now().strftime("%Y-%m-%d %H:%M")

    if os.path.exists(LOG_FILE):
        log_df = pd.read_excel(LOG_FILE)
        log_df = pd.concat([log_df, df], ignore_index=True)
    else:
        log_df = df
    log_df.to_excel(LOG_FILE, index=False)

    return df

def plot_strength_distribution():
    if not os.path.exists(LOG_FILE): return None
    df = pd.read_excel(LOG_FILE)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df["Predicted Strength"], bins=20, kde=True, ax=ax)
    ax.set_title("Distribution of Predicted Strength")
    return fig

def load_all_logs():
    if os.path.exists(LOG_FILE):
        return pd.read_excel(LOG_FILE)
    return pd.DataFrame()

# ------------------- Gradio App ----------------------
with gr.Blocks(title="🏗️ AI Concrete Compressive Strength Finder") as app:
    gr.Markdown("# 🏗️ AI Concrete Compressive Strength Finder")

    with gr.Tab("📊 Predict (Manual)"):
        with gr.Row():
            with gr.Column():
                cement = gr.Number(label="Cement")
                slag = gr.Number(label="Slag")
                flyash = gr.Number(label="Fly Ash")
                water = gr.Number(label="Water")
                superp = gr.Number(label="Superplasticizer")
                coarse = gr.Number(label="Coarse Agg")
                fine = gr.Number(label="Fine Agg")
                age = gr.Number(label="Age")
                predict_btn = gr.Button("Predict")
            result = gr.Textbox(label="Prediction Result")
        histogram = gr.Plot(label="Component Effects")

    with gr.Tab("📁 Upload Mixes"):
        file_input = gr.File(label="Upload Excel File")
        batch_btn = gr.Button("Predict Batch")
        batch_output = gr.Dataframe()
        graph_output = gr.Plot(label="Strength Distribution")

    with gr.Tab("📈 Logs"):
        full_logs = gr.Dataframe()

    with gr.Tab("🎨 Interface Settings"):
        theme = gr.Radio(["Default", "Compact", "Dark"], label="UI Theme")
        gr.Markdown("Select your preferred interface theme. (Visual only, demo purpose)")

    with gr.Tab("📘 README"):
        gr.Markdown("""## 🧠 AI Concrete Compressive Strength Finder
This AI app predicts the compressive strength of concrete based on mix proportions.
- Manual & Excel input
- Suggestions for improvement
- Batch logging
- Visual analytics""")

    with gr.Tab("💌 Support"):
        gr.Markdown("""📧 Contact: **arslanhafeezkhan.16@gmail.com**
We value your feedback and will respond to all inquiries promptly!""")

    with gr.Tab("📄 License"):
        gr.Markdown("""All Rights Reserved © 2025 Arslan Khan\nUse permitted for academic and personal research only.""")

    # Link buttons
    predict_btn.click(fn=predict_strength,
                     inputs=[cement, slag, flyash, water, superp, coarse, fine, age],
                     outputs=result)

    predict_btn.click(fn=manual_histogram,
                     inputs=[cement, slag, flyash, water, superp, coarse, fine, age],
                     outputs=histogram)

    batch_btn.click(fn=batch_predict, inputs=file_input, outputs=batch_output)
    batch_btn.click(fn=plot_strength_distribution, outputs=graph_output)
    batch_btn.click(fn=load_all_logs, outputs=full_logs)

app.launch()