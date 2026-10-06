# 🌊 Coral Reef Regime Predictor

This project is a multiclass machine learning classification system designed to predict coral reef regimes based on 20 distinct anthropogenic, biophysical, and environmental features. The best performing model (Random Forest) has been extracted and deployed through a Streamlit web application.

## 🚀 Live Demo
You can view the live deployment of this application at: **https://coral-reef-regime.streamlit.app/**

---

## 🛠️ Local Setup Instructions

If you would like to run this application locally on your own machine, follow the instructions below.

### 1. Clone the Repository
```bash
git clone https://github.com/diya-122/Coral-Reef-Regime.git
cd Coral-Reef-Regime
```

### 2. Create a Virtual Environment (Optional but Recommended)
**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```
**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install the required Python packages using the provided `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
Launch the web interface by executing the following command:
```bash
python -m streamlit run app.py
```
*The application should automatically open in your default web browser at `http://localhost:8501`.*

---

## 📂 Repository Structure
* `app.py`: The main Streamlit web application.
* `train_and_save.py`: Script used to download the dataset, train the Random Forest model, and save the pickle files.
* `requirements.txt`: List of dependencies required to run the project.
* `models/`: Contains the trained `best_model.pkl`, the `features.pkl` for correct ordering, and the Jupyter Notebooks documenting the research, EDA, feature engineering, and model comparison phases.
* `poster.pdf` & `research paper.pdf`: Reference documents and project presentation materials.
