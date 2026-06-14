"""
Flask Web Application — Diabetes Risk Prediction
Backend: loads trained Voting Classifier, preprocesses input, returns prediction.
"""

import os, sys
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')

app = Flask(__name__)

# ── Load artifacts once at startup ───────────────────────────────────────────
try:
    model         = joblib.load(os.path.join(MODELS_DIR, 'voting_classifier.pkl'))
    scaler        = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    encoders      = joblib.load(os.path.join(MODELS_DIR, 'encoders.pkl'))
    feature_names = joblib.load(os.path.join(MODELS_DIR, 'feature_names.pkl'))
    cat_cols      = joblib.load(os.path.join(MODELS_DIR, 'cat_cols.pkl'))
    num_feat      = joblib.load(os.path.join(MODELS_DIR, 'num_feat.pkl'))
    print("[INFO] All model artifacts loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load model artifacts: {e}")
    model = None

CLASS_LABELS = {0: 'Non-Diabetic', 1: 'Pre-Diabetic', 2: 'Diabetic'}
CLASS_COLORS = {0: '#2ecc71', 1: '#f39c12', 2: '#e74c3c'}
CLASS_RISK   = {
    0: 'Low Risk — No diabetes markers detected. Maintain healthy lifestyle.',
    1: 'Moderate Risk — Pre-diabetic indicators present. Lifestyle changes recommended.',
    2: 'High Risk — Diabetic indicators detected. Immediate medical consultation advised.'
}

# ── Default field values (median/mode of dataset) ────────────────────────────
DEFAULTS = {
    'age':                    '[60-70)',
    'race':                   'Caucasian',
    'gender':                 'Male',
    'time_in_hospital':       7,
    'num_lab_procedures':     44,
    'num_procedures':         1,
    'num_medications':        15,
    'number_outpatient':      0,
    'number_emergency':       0,
    'number_inpatient':       0,
    'number_diagnoses':       8,
    'glucose_level':          140.0,
    'hba1c_level':            7.5,
    'admission_type_id':      1,
    'discharge_disposition_id': 1,
    'admission_source_id':    7,
}


def build_feature_vector(form_data):
    """
    Convert raw form input into the feature vector expected by the model.
    Handles encoding, imputation for missing features, and scaling.
    """
    # Start with a default row for all features
    row = {feat: 0 for feat in feature_names}

    # Fill numerical fields from form
    num_fields = {
        'time_in_hospital':       float(form_data.get('time_in_hospital', DEFAULTS['time_in_hospital'])),
        'num_lab_procedures':     float(form_data.get('num_lab_procedures', DEFAULTS['num_lab_procedures'])),
        'num_procedures':         float(form_data.get('num_procedures', DEFAULTS['num_procedures'])),
        'num_medications':        float(form_data.get('num_medications', DEFAULTS['num_medications'])),
        'number_outpatient':      float(form_data.get('number_outpatient', DEFAULTS['number_outpatient'])),
        'number_emergency':       float(form_data.get('number_emergency', DEFAULTS['number_emergency'])),
        'number_inpatient':       float(form_data.get('number_inpatient', DEFAULTS['number_inpatient'])),
        'number_diagnoses':       float(form_data.get('number_diagnoses', DEFAULTS['number_diagnoses'])),
        'glucose_level':          float(form_data.get('glucose_level', DEFAULTS['glucose_level'])),
        'hba1c_level':            float(form_data.get('hba1c_level', DEFAULTS['hba1c_level'])),
        'admission_type_id':      float(form_data.get('admission_type_id', DEFAULTS['admission_type_id'])),
        'discharge_disposition_id': float(form_data.get('discharge_disposition_id', DEFAULTS['discharge_disposition_id'])),
        'admission_source_id':    float(form_data.get('admission_source_id', DEFAULTS['admission_source_id'])),
    }
    for k, v in num_fields.items():
        if k in row:
            row[k] = v

    # Fill categorical fields using stored encoders
    cat_fields = {
        'age':    form_data.get('age',    DEFAULTS['age']),
        'race':   form_data.get('race',   DEFAULTS['race']),
        'gender': form_data.get('gender', DEFAULTS['gender']),
    }
    for col, val in cat_fields.items():
        if col in encoders and col in row:
            try:
                row[col] = int(encoders[col].transform([val])[0])
            except Exception:
                # Unseen label — use 0
                row[col] = 0

    # Build ordered feature vector (use DataFrame to preserve feature names → suppresses sklearn warning)
    X_raw = pd.DataFrame([[row[f] for f in feature_names]], columns=feature_names, dtype=float)
    X_scaled = scaler.transform(X_raw)
    return X_scaled


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Run pipeline.py first.'}), 503

    try:
        form_data = request.form.to_dict()
        X = build_feature_vector(form_data)

        pred_class  = int(model.predict(X)[0])
        prob_array  = model.predict_proba(X)[0]
        confidence  = float(prob_array[pred_class]) * 100

        result = {
            'class_id':    pred_class,
            'class_label': CLASS_LABELS[pred_class],
            'confidence':  round(confidence, 1),
            'color':       CLASS_COLORS[pred_class],
            'risk_message': CLASS_RISK[pred_class],
            'probabilities': {
                'Non-Diabetic': round(float(prob_array[0]) * 100, 1),
                'Pre-Diabetic': round(float(prob_array[1]) * 100, 1),
                'Diabetic':     round(float(prob_array[2]) * 100, 1),
            }
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
