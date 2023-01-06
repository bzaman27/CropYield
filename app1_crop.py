# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 18:44:30 2023

@author: Bushra
"""

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
    <h2 style="color:white;text-align:center;">Crop Yield Predictor ML App </h2>
    <head>
    <meta charset="UTF-8">
  <meta http-equiv="X-UA-Compatible" content="IE=edge">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
	<title>DCAP</title>
	<link rel="shortcut icon" href="static/images/lsmall.jpg"/>

	<script>
		addEventListener("load", function () {
			setTimeout(hideURLbar, 0);
		}, false);

		function hideURLbar() {
			window.scrollTo(0, 1);
		}

	</script>

	<script src="https://code.jquery.com/jquery-3.3.1.slim.min.js"
		integrity="sha384-q8i/X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4YfRvH+8abtTE1Pi6jizo"
		crossorigin="anonymous"></script>
	<script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.7/umd/popper.min.js"
		integrity="sha384-UO2eT0CpHqdSJQ6hJty5KVphtPhzWj9WO1clHTMGa3JDZwrnQq4sF86dIHNDz0W1"
		crossorigin="anonymous"></script>
	<script src="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/js/bootstrap.min.js"
		integrity="sha384-JjSmVgyd0p3pXB1rRibZUAYoIIy6OrQ6VrjIEaFf/nJGzIxFDsf4x0xIM+B07jRM"
		crossorigin="anonymous"></script>
	<script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"
		integrity="sha384-DfXdz2htPH0lsSSs5nCTpuj/zy4C+OGpamoFVy38MVBnE+IbbVYUew+OrCXaRkfj"
		crossorigin="anonymous"></script>
	<script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.0/dist/umd/popper.min.js"
		integrity="sha384-Q6E9RHvbIyZFJoft+2mJbHaEWldlvI9IOYy5n3zV9zzTtmI3UksdQRVvoxMfooAo"
		crossorigin="anonymous"></script>
	<!-- css files -->
	<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css"
		integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">
	<link href="{{ url_for('static', filename='css/bootstrap.css') }}" rel='stylesheet' type='text/css' />
	<!-- bootstrap css -->
	<link href="{{ url_for('static', filename='css/style.css') }}" rel='stylesheet' type='text/css' />
	<!-- custom css -->
	<link href="{{ url_for('static', filename='css/font-awesome.min.css') }}" rel="stylesheet"><!-- fontawesome css -->
	<!-- //css files -->
	<!-- <link rel="icon" type="image/png" href="{{ url_for('static', filename='images/favicon.png?') }}"> -->


	<script type="text/JavaScript" src="{{ url_for('static', filename='scripts/cities.js') }}"></script>


	<!-- google fonts -->
	<link href="//fonts.googleapis.com/css?family=Thasadith:400,400i,700,700i&amp;subset=latin-ext,thai,vietnamese"
		rel="stylesheet">
	<!-- //google fonts -->

	<style>
        body {
            background-color: white;
            background-image: url(static/images/crop_background.jpg);
            background-size: 100% 100%;
            text-align: center;
            padding: 0px;
        }

        #sub {
            width: 300px;
            height: 45px;
            text-align: center;
            border-radius: 14px;
            font-size: 18px;
            transition: 0.3s;
        }

            #sub:hover {
                background-color: darkcyan;
                color: white;
            }

        #first, #second, #third, #fourth {
            border-radius: 14px;
            padding: 2px;
            width: 250px;
            font-size: 16px;
            text-align: center;
        }
	</style>
</head>

<body>
    <!-- Navigation -->
    <nav class="navbar navbar-expand-lg navbar-dark static-top" style="background-color: black">
        <div class="container">
            <a class="navbar-brand" href="{{ url_for('Crop_Yield') }}">
                <img src='static/images/logo.jpg' style="width: 200px; height: 40px;" alt="">
            </a>
            <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarResponsive"
                    aria-controls="navbarResponsive" aria-expanded="false" aria-label="Toggle navigation">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarResponsive">
                <ul class="navbar-nav ml-auto">
                    <li class="nav-item active">
                        <a class="nav-link" href="{{ url_for('Crop_Yield') }}">
                            Crop_Yield
                            <span class="sr-only">(current)</span>
                        </a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <!-- footer -->
    <footer class="text-center">
        <div class="container mt-3">
            <!-- logo -->
            <h2 class="logo2 text-center">
                <a style="color: rgb(0, 70, 176);" href="{{ url_for('Crop_Yield') }}">
                    DSAI DIGI_AGRI PLATFORM
                </a>
            </h2>
            <!-- //logo -->
            <!-- address -->
            <div class="contact-left-footer mt-4">

                <!-- <a href="community.html">Community</a> -->
                <!-- </p> -->
            </div>
            <div class="w3l-copy text-center">
                <p class="text-da">A SDG Initiative<br> </p>
            </div>
            <p>&copy; Copyright 2022 DSAI</p>
        </div>
    </footer>
    <div style="color:rgb(87, 134, 255);">
        <br action="{{ url_for('predict')}}" method="post">
            <h3>Predictive analytics for Crop Yield </h3>
            <h6 style="margin-top: 10px;">Area (in hectares)</h6><input id="first" name="Area" type="number" style="border: 1.5px solid black">
            <h6 style="margin-top: 10px;">What is the Temperature (in degree celcius)?</h6><input id="second" name="Temperature" required="required" style="border: 1.5px solid black">
            <h6 style="margin-top: 10px;">How much is the Precipitation (in mm)?</h6><input id="third" name="Precipitation" required="required" style="border: 1.5px solid black">
            <h6 style="margin-top: 10px;">How much is the Humidity (in grams per cubic meter)?</h6><input id="fourth" name="Humidity" required="required" style="border: 1.5px solid black">
            <br></br><button id="sub" type="submit ">Calculate the Crop Yield</button>
            <br>
        </form>
        <br><br><h3>{{ prediction_text }}<h3>
    </div>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    Area = st.text_input("Area in hectares","Enter value")
    Temperature = st.text_input("Temperature in degree celcius","Enter value")
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
    
    
    