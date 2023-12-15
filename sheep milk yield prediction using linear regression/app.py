from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import shap

from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import shap

app = Flask(__name__)
# Read the CSV file into a pandas DataFrame
df = pd.read_csv('sheep_milk_yield_encoded.csv')  # Replace 'path_to_your_file.csv' with the actual path to your CSV file
# Assuming 'df' is your DataFrame containing the data
# Select features (X) and target variable (y)
features = ['Age', 'D1WT', 'Breed']
X = df[features]
y = df['D1MILK']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (optional but recommended for linear regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Load the pre-trained model and scaler
model = LinearRegression()
scaler = StandardScaler()

# Assuming 'X_train_scaled' and 'y_train' are available from your training process
# 'model' should be your trained linear regression model
# 'features' should be the list of features used in training
# 'explainer' is used to generate SHAP values
X_train_scaled = scaler.fit_transform(X_train)
model.fit(X_train_scaled, y_train)
explainer = shap.Explainer(model, X_train_scaled)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = float(request.form['age'])
        d1wt = float(request.form['d1wt'])
        breed = int(request.form['breed'])

        # Standardize input data
        new_data = pd.DataFrame({'Age': [age], 'D1WT': [d1wt], 'Breed': [breed]})
        new_data_scaled = scaler.transform(new_data)

        # Make prediction
        predicted_milk_yield = model.predict(new_data_scaled)[0]

        # Generate SHAP values for the prediction
        shap_values = explainer.shap_values(new_data_scaled)[0]

        # Pass the prediction and SHAP values to the template
        return render_template('result.html', prediction=predicted_milk_yield, shap_values=shap_values)

if __name__ == '__main__':
    app.run(debug=True)
