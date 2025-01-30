import os
import numpy as np
import joblib
import pandas as pd
from flask import Flask, render_template, request
from sklearn.preprocessing import MinMaxScaler

# Initialize Flask app
app = Flask(__name__)

# Get the base directory (works on Render)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Function to load models and scalers safely
def load_model_and_scalers(model_name):
    try:
        model_path = os.path.join(BASE_DIR, f'{model_name}.pkl')
        scaler_X_path = os.path.join(BASE_DIR, f'{model_name}_minmax_scaler_X.pkl')
        scaler_y_path = os.path.join(BASE_DIR, f'{model_name}_minmax_scaler_y.pkl')

        if not os.path.exists(model_path):
            print(f"⚠️ Model file not found: {model_path}")
            return None, None, None
        if not os.path.exists(scaler_X_path):
            print(f"⚠️ Scaler X file not found: {scaler_X_path}")
            return None, None, None
        if not os.path.exists(scaler_y_path):
            print(f"⚠️ Scaler Y file not found: {scaler_y_path}")
            return None, None, None

        model = joblib.load(model_path)
        scaler_X = joblib.load(scaler_X_path)
        scaler_y = joblib.load(scaler_y_path)
        return model, scaler_X, scaler_y
    except Exception as e:
        print(f"❌ Error loading {model_name}: {e}")
        return None, None, None


# Load Models and Scalers
models = {
    'Beam-Slab': {
        'Volume of Concrete': load_model_and_scalers('BeamSlab - Trial 4.1 Volume of Concrete_iteration8_checking_curves'),
        'Volume of Reinforcement': load_model_and_scalers('BeamSlab - Trial 4.2 Volume of Reinforcement_iteration7_Curves'),
        'Embodied Carbon Energy': load_model_and_scalers('BeamSlab - Trial 4.3 Embodied Carbon Energy (GJ)_iteration7_Curves'),
        'Emissions CO2 GWP': load_model_and_scalers('BeamSlab - Trial 4.4 Emissions CO2 GWP_iteration7_Curves')
    },
    'Flat-Slab': {
        'Volume of Concrete': load_model_and_scalers('FlatSlab - Trial 4.1 Volume of Concrete_iteration7'),
        'Volume of Reinforcement': load_model_and_scalers('FlatSlab - Trial 4.2 Volume of Reinforcement_iteration8_Curves'),
        'Embodied Carbon Energy': load_model_and_scalers('FlatSlab - Trial 4.3 Embodied Carbon Energy (GJ)_iteration8_Curves'),
        'Emissions CO2 GWP': load_model_and_scalers('FlatSlab - Trial 4.4 Emissions CO2 GWP_iteration8_Curves')
    }
}

# Load CSV database files safely
file_path_beam_slab = os.path.join(BASE_DIR, 'Database_BeamSlab_Changed Usage-names updated.csv')
file_path_flat_slab = os.path.join(BASE_DIR, 'FlatSlab Database-names updated.csv')

try:
    df_beam_slab = pd.read_csv(file_path_beam_slab, encoding='latin-1')
except Exception as e:
    df_beam_slab = None
    print(f"⚠️ Error loading Beam-Slab CSV: {e}")

try:
    df_flat_slab = pd.read_csv(file_path_flat_slab, encoding='latin-1')
except Exception as e:
    df_flat_slab = None
    print(f"⚠️ Error loading Flat-Slab CSV: {e}")


# Convert inputs to features
def convert_inputs_to_features(span_length_1, span_length_2, usage, boundary_condition, floor_system, model_type):
    usage_map = {'Residential': [1, 0, 0], 'Common space': [0, 1, 0], 'Office': [0, 0, 1]}
    boundary_condition_map = {'Internal Panel': [1, 0, 0], 'Edge Panel': [0, 1, 0], 'Corner Panel': [0, 0, 1]}
    floor_system_map = {
        'One-way': [1, 0, 0, 0],
        'Two-way': [0, 1, 0, 0],
        'One-way with Secondary beam': [0, 0, 1, 0],
        'Two-way with Secondary beam': [0, 0, 0, 1]
    }
    usage_onehot = usage_map.get(usage, [0, 0, 0])
    boundary_condition_onehot = boundary_condition_map.get(boundary_condition, [0, 0, 0])
    floor_system_onehot = floor_system_map.get(floor_system, [0, 0, 0, 0]) if model_type == 'Beam-Slab' else []

    return np.array([span_length_1, span_length_2] + usage_onehot + boundary_condition_onehot + floor_system_onehot)


# Route for the main page
@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = None  # Initialize predictions

    if request.method == 'POST':
        try:
            # Get form data
            span_length_1 = float(request.form['span-length-1'])
            span_length_2 = float(request.form['span-length-2'])
            usage = request.form['usage']
            boundary_condition = request.form['boundary-condition']
            floor_system = request.form['floor-system']
            model_type = request.form['model-type']

            # Convert inputs to features
            features = convert_inputs_to_features(span_length_1, span_length_2, usage, boundary_condition, floor_system, model_type)

            predictions = {}
            for target, (model, scaler_X, scaler_y) in models[model_type].items():
                if model and scaler_X and scaler_y:
                    try:
                        features_scaled = scaler_X.transform(features.reshape(1, -1))
                        pred = model.predict(features_scaled)
                        predictions[target] = scaler_y.inverse_transform(pred.reshape(-1, 1))[0][0]
                    except Exception as e:
                        predictions[target] = f"Error with {target}: {str(e)}"
                else:
                    predictions[target] = 'Model/Scaler Not Loaded'

        except Exception as e:
            print(f"❌ Error processing form data: {e}")

    return render_template("index.html", predictions=predictions)  # Remove database tables from the template


# Run Flask app (Render will set PORT automatically)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render sets PORT dynamically
    app.run(debug=False, host='0.0.0.0', port=port)

