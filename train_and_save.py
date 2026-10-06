import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Paths
BASE_URL = "https://raw.githubusercontent.com/salliewalecka/Hawaii_RegimesPredictors/master/Data/Modeling/"
TRAIN_FILE = "Predictors_complete_train.txt"

def train_and_save_model():
    print("Downloading dataset...")
    df = pd.read_csv(BASE_URL + TRAIN_FILE, sep="\t", decimal=",")
    
    # Target
    y = df['Regime']
    
    # 20 Base + 5 Engineered features
    features = [
        'Effluent', 'Sedimentation', 'New_Development', 'Habitat_Modification', 'Invasive_Algae',
        'Fishing_Comm_Total', 'Fishing_NonComm_Boat_Total', 'Fishing_NonComm_Shore_Line', 
        'Fishing_NonComm_Shore_Net', 'Fishing_NonComm_Shore_Spear', 
        'SST_CLIM_M', 'SST_STD', 'CHL_CLIM_M', 'CHL_ANOM_F', 'PAR_CLIM_M', 'PAR_STD', 
        'WAV_CLIM_M', 'WAV_ANOM_F', 'Complexity', 'Depth',
        'sqrt_power_x_compexity', 'log_power_over_depth', 'complexity_over_depth', 
        'Irradiance_x_inv_algae', 'both_anomolies'
    ]
    
    X = df[features]
    
    print("Training Random Forest model...")
    # Best params can be generic or default if exact ones are unknown
    model = RandomForestClassifier(n_estimators=100, random_state=229)
    model.fit(X, y)
    
    if not os.path.exists('models'):
        os.makedirs('models')
        
    print("Saving model to models/best_model.pkl...")
    joblib.dump(model, 'models/best_model.pkl')
    # Save the feature list for Streamlit reference
    joblib.dump(features, 'models/features.pkl')
    
    print("Done!")

if __name__ == "__main__":
    train_and_save_model()
