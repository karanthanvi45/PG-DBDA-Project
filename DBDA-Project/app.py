from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
with open('rfc2.pkl', 'rb') as file:
    model = pickle.load(file)

def convert_to_int(value, default=0):
    try:
        return int(value)
    except ValueError:
        return default

def convert_to_float(value, default=0.0):
    try:
        return float(value)
    except ValueError:
        return default

@app.route('/')
def index():
    return render_template('index.html', tab="predict")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Collect the input features from the form
        features = [
            convert_to_int(request.form['age']),
            convert_to_int(request.form['job']),
            convert_to_int(request.form['marital']),
            convert_to_int(request.form['education']),
            convert_to_int(request.form['default']),
            convert_to_int(request.form['housing']),
            convert_to_int(request.form['loan']),
            convert_to_int(request.form['contact']),
            convert_to_int(request.form['month']),
            convert_to_int(request.form['day_of_week']),
            convert_to_int(request.form['poutcome']),
            convert_to_int(request.form['campaign']),
            convert_to_int(request.form['pdays']),
            convert_to_int(request.form['previous']),
            convert_to_float(request.form['emp_var_rate']),
            convert_to_float(request.form['cons_price_idx']),
            convert_to_float(request.form['cons_conf_idx']),
            convert_to_float(request.form['euribor3m']),
            convert_to_float(request.form['nr_employed'])
        ]
        
        final_features = np.array(features).reshape(1, -1)
        prediction = model.predict(final_features)
        
        output = "YES" if prediction[0] == 1 else "NO"
        
        return render_template('index.html', prediction_text=f'Predicted Outcome: {output}', tab="predict")

@app.route('/visualization')
def visualization():
    return render_template('index.html', tab="visualization")

if __name__ == "__main__":
    app.run(debug=True)

