from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

# # Load the model and scaler
# with open('../model/best_model.pkl', 'rb') as f:
#     data = pickle.load(f)
#     model = data['model']
#     scaler = data['scaler']
import os

# Build the absolute path to best_model.pkl relative to this file
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
model_path = os.path.join(base_dir, 'model', 'best_model.pkl')

from joblib import load
data = load(model_path)
model = data['model']
scaler = data['scaler']



# Home route
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            # Retrieve form data
            # Note: Ensure the names in the HTML form match these keys.
            CreditScore = float(request.form['CreditScore'])
            Age = float(request.form['Age'])
            Tenure = float(request.form['Tenure'])
            Balance = float(request.form['Balance'])
            NumOfProducts = float(request.form['NumOfProducts'])
            HasCrCard = float(request.form['HasCrCard'])
            IsActiveMember = float(request.form['IsActiveMember'])
            EstimatedSalary = float(request.form['EstimatedSalary'])
            # For categorical variables, we assume the form sends the already encoded values:
            # For Geography: assume form sends either "Germany" or "Spain" (if not, default is France)
            Geography = request.form.get('Geography', 'France')
            # For Gender: assume form sends either "Male" or "Female" (if not, default is Female)
            Gender = request.form.get('Gender', 'Female')

            # Create a feature array in the order expected by the model.
            # Our training data after preprocessing had:
            # CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary,
            # Geography_Spain, Geography_Germany, Gender_Male
            Geography_Spain = 1 if Geography.lower() == 'spain' else 0
            Geography_Germany = 1 if Geography.lower() == 'germany' else 0
            Gender_Male = 1 if Gender.lower() == 'male' else 0

            features = np.array([[CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary,
                                  Geography_Spain, Geography_Germany, Gender_Male]])

            # Scale the features using the loaded scaler
            features_scaled = scaler.transform(features)
            pred = model.predict(features_scaled)
            prediction = "Customer will leave" if pred[0] == 1 else "Customer will stay"
        except Exception as e:
            prediction = f"Error in prediction: {str(e)}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
