# 🛡️ Network Intrusion Detection System (NIDS) using Machine Learning

A machine learning-based web application to detect network intrusions in real time. This project helps identify various types of attacks such as DoS, Probe, R2L, and U2R using a trained classification model. The system also features a user-friendly dashboard for uploading traffic data, visualizing threats, and monitoring alerts.

---
🎥 Project Video
▶️ https://drive.google.com/file/d/16uqHMlLWNNATcbE761KgJPsCIsYvG0Cg/view?usp=drivesdk
## 📌 Features

- ✅ Upload network traffic datasets in CSV format
- ✅ Real-time prediction of threats using ML model
- ✅ Visualization of detected threats with live graphs
- ✅ Threat summary analytics (counts, percentages, categories)
- ✅ Alerts panel for immediate attention
- ✅ Flask-based backend integrated with trained model

---

## 📊 Screens and Pages

### 1. **Home Page**
- Introduction to the system
- How the model works and its purpose

### 2. **Upload Page**
- Upload test data (CSV)
- Backend processes and predicts threats using ML model

### 3. **Results Page**
- Tabular display of records with predicted labels
- Clear identification of attack types

### 4. **Real-Time Threat Detection Graph**
- Line or bar graph showing detected threats over time
- Helps monitor spikes in network attacks

### 5. **Analytics Page**
- Total records processed
- Number and type of threats detected
- Visual stats like pie charts or bar charts

### 6. **Alerts Page**
- Immediate listing of recent threats
- Timestamps and categories for each alert

---

## 🧠 Machine Learning Model

- **Dataset Used:** NSL-KDD Dataset
- **Algorithms Tried:** Random Forest, Decision Tree, SVM
- **Final Model:** Random Forest (best performance in accuracy and recall)
- **Preprocessing Steps:**
  - Label encoding for categorical features
  - Feature scaling/normalization
  - Attack type classification
- **Metrics Evaluated:** Accuracy, Precision, Recall, F1-Score

---

## 🛠️ Tech Stack

| Component       | Technology               |
|----------------|--------------------------|
| Frontend        | HTML, CSS, Bootstrap     |
| Backend         | Python (Flask)           |
| ML Model        | Scikit-learn             |
| Data Handling   | Pandas, NumPy            |
| Visualization   | Matplotlib / Plotly      |

---

## 🚀 How to Run the Project Locally

```bash
# 1. Clone the repository
git clone https://github.com/your-username/nids-ml-webapp.git
cd nids-ml-webapp

# 2. Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Flask server
python app.py

# 5. Open your browser and go to:
http://127.0.0.1:5000/
