# Banking Customer Churn Prediction

This project builds a machine learning model to predict customer churn in the banking sector and provides a web application for bankers to input customer data and receive predictions.

## Project Structure

BankingChurnPrediction/ ├── app/ │ ├── static/ │ │ └── css/ │ │ └── style.css │ ├── templates/ │ │ └── index.html │ └── app.py ├── model/ │ └── best_model.pkl ├── notebooks/ │ └── ChurnPredictionModel.ipynb ├── requirements.txt └── README.md


## Steps to Run

1. **Model Training:**
   - Open the Jupyter Notebook `notebooks/ChurnPredictionModel.ipynb`.
   - Run all cells to preprocess data, train different models, evaluate them, and save the best model to `model/best_model.pkl`.

2. **Web Application:**
   - Install dependencies with `pip install -r requirements.txt`.
   - Run the Flask app:
     
     cd app
     python app.py
     
   - Open your browser and go to `http://127.0.0.1:5000` to access the application.

## Hosted on Render.com
  - URL: ``

## About

This application uses machine learning for churn prediction to help banks identify customers who may leave, enabling targeted retention strategies.

**Built by Sayandeep Dutta**

