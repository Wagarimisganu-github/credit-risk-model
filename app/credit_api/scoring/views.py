# Create your views here.
from rest_framework.decorators import api_view
from rest_framework.response import Response
import joblib
import numpy as np
import os
from django.shortcuts import render
from .forms import PredictionForm
import os, sys
sys.path.append(os.path.abspath(os.path.join('..')))
#from ../scripts ml_preprocess import create_preprocessing_pipeline, extract_date_features
#from scripts.ml_preprocess import *
import pandas as pd

from .utils import *


# Load the saved model
model_path = os.path.join(os.path.dirname(__file__), './model/best_credit_scoring_model.pkl')
model = joblib.load(model_path)
#credit_api\scoring\model\best_credit_scoring_model.pkl

# Define an API to serve predictions
@api_view(['POST'])
def predict_credit_risk(request):
    data = request.data.get('features', None)
    
    if data:
        # Create DataFrame from the input features (Assuming input as a dictionary of lists)
        df = pd.DataFrame([data])
        
        # Apply the preprocessing steps
        df = handle_missing_values(df)
        categorical_columns = ['ProductCategory', 'ChannelId', 'PricingStrategy', 'FraudResult']
        df = encode_categorical_features(df, categorical_columns)
        df = normalize_features(df)
        
        # Make prediction
        features = df.to_numpy()  # Convert preprocessed DataFrame to numpy array
        prediction = model.predict(features)
        return Response({'prediction': prediction.tolist()})
    else:
        return Response({'error': 'No data provided'}, status=400)

# UI for submitting data
def home(request):
    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            # Extract form values into a DataFrame
            data = {field: form.cleaned_data[field] for field in form.fields}
            df = pd.DataFrame([data])
            
            # Apply preprocessing
            df = handle_missing_values(df)
            categorical_columns = ['ProductCategory', 'ChannelId', 'PricingStrategy', 'FraudResult']
            df = encode_categorical_features(df, categorical_columns)
            #df = normalize_features(df)
            
            # Make prediction
            features = df.to_numpy()
            prediction = model.predict(features)
            return render(request, 'scoring/home.html', {'form': form, 'prediction': prediction[0]})
    else:
        form = PredictionForm()
    
    return render(request, 'scoring/home.html', {'form': form, 'prediction': None})