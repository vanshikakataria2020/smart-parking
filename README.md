# 🅿️ Smart Parking Availability Prediction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.18%2B-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**An AI-powered smart city dashboard that predicts parking occupancy levels in real time using a RandomForest Classifier.**

[🚀 Live Demo](#) &nbsp;|&nbsp; [📂 Dataset](#dataset) &nbsp;|&nbsp; [🛠️ Installation](#installation) &nbsp;|&nbsp; [📊 Features](#features)

</div>

---

## 📌 Overview

A Machine Learning project built for an **AAT (Assignment/Assessment Task)** at BMS College of Engineering. The system predicts whether a parking facility is at **Low**, **Medium**, or **High** occupancy based on parking analytics data.

The dashboard is built with **Streamlit** and styled as a smart city analytics interface, featuring live input controls, interactive charts, and ML-powered predictions.

---

## ✨ Features

- 🤖 **ML Prediction** — RandomForestClassifier predicts occupancy level (Low / Medium / High)
- 📊 **Interactive Dashboard** — Real-time metric cards, pie chart, gauge, trend graph, feature importance
- 🎛️ **Live Input Controls** — Adjust slots, traffic, revenue, and time parameters on the fly
- 🌙 **Dark UI** — Professional dark smart-city aesthetic with Plotly visualizations
- 📱 **Responsive Layout** — Works on desktop and tablet browsers
- 🔌 **Demo Mode** — Runs with a surrogate model if no `.pkl` file is present

---

## 🧠 Model Details

| Property | Details |
|---|---|
| Algorithm | `RandomForestClassifier` (scikit-learn) |
| Training Environment | Google Colab |
| Output Classes | `Low` · `Medium` · `High` |
| Input Features | 9 (see below) |

### Input Features

| Feature | Description |
|---|---|
| `total_slots` | Total parking capacity of the facility |
| `occupied_slots` | Number of currently occupied slots |
| `avg_parking_duration_minutes` | Average time a vehicle stays parked |
| `entry_count` | Number of vehicles that entered |
| `exit_count` | Number of vehicles that exited |
| `parking_fee_collected` | Total fee collected (₹) |
| `Hour` | Hour of the day (0–23) |
| `Day` | Day of the week (0=Mon, 6=Sun) |
| `Month` | Month of the year (1–12) |

---

## 📊 Dashboard Sections

- **Hero Banner** — Project title, description, and AI badge
- **Live Parking Overview** — 4 metric cards (Total / Occupied / Available Slots + Occupancy %)
- **Input Controls** — Expandable panel with all 9 input features
- **Prediction Result** — Color-coded result card (🟢 Low · 🟡 Medium · 🔴 High)
- **Occupancy Pie Chart** — Probability distribution across classes
- **Occupancy Gauge** — Real-time fill rate speedometer
- **24-Hour Trend Graph** — Occupancy pattern across the day
- **Feature Importance Chart** — Which features drive the prediction most
- **Model Info Strip** — Algorithm, framework, and feature count summary

---

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/vanshikakataria2020/smart-parking-ai.git
cd smart-parking-ai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your trained model *(optional)*
```bash
# Place your trained model file in the project root
cp /path/to/your/parking_model.pkl .
```
> If no model file is found, the app runs in **Demo Mode** using a surrogate RandomForest trained on synthetic data.

### 4. Run the app
```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| Frontend / UI | Streamlit, Custom CSS |
| ML Model | scikit-learn RandomForestClassifier |
| Visualization | Plotly |
| Model Serialization | Pickle |
| Training Environment | Google Colab |
| Deployment | Streamlit Community Cloud |

---

## 📁 Project Structure

```
smart-parking-ai/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── parking_model.pkl       # Trained ML model (add your own)
└── README.md               # Project documentation
```

---

## 🚀 Deployment

This app is deployed on **Streamlit Community Cloud**.

👉 **Live Link:** https://smart-parking-vanshika.streamlit.app/ *

To deploy your own:
1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file to `app.py` → click **Deploy**

---

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

**Developed for Smart Parking Availability Prediction System**  
*Machine Learning · Streamlit · AAT Project*

⭐ Star this repo if you found it helpful!

</div>
