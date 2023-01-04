# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 14:43:31 2023

@author: Bushra
"""

#from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd
from PIL import Image
import jsonify
import requests
import sklearn
from sklearn.preprocessing import StandardScaler
import streamlit as st

#app = Flask(__name__)
model = pickle.load(open('random_forest_regression_model.pkl', 'rb'))
#@app.route('/',methods=['GET'])
def welcome():
    return "Welcome All"

def predict_crop_yield(Area,Temperature,Precipitation,Humidity):
    
    """Let's Predict the Crop Yield 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: Area
        in: query
        type: number
        required: true
      - name: Temperature
        in: query
        type: number
        required: true
      - name: Precipitation
        in: query
        type: number
        required: true
      - name: Humidity
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
   
    prediction=model.predict([[Area,Temperature,Precipitation,Humidity]])
    print(prediction)
    return prediction

    
def main():
    st.title("Crop Yield Prediction")
    html_temp = """
    <div style="background-color:green;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Crop Yield Predictor ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    Area = st.text_input("Area in hectares","Enter value")
    Temperature = st.text_input("Temperature in degree celcius","TEnter value")
    Precipitation = st.text_input("Precipitation in mm","Enter value")
    Humidity = st.text_input("Humidity in percent","Enter value")
    result=""
    if st.button("Predict"):
        result=predict_crop_yield(Area,Temperature,Precipitation,Humidity)
    st.success('The output is [ {} ] kg/hectare'.format(result))
    if st.button("About"):
        st.text("Lets Learn")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()
    
    
    