import json

import requests
from django.shortcuts import render
from .models import MobilePriceForm
import joblib
import numpy as np
import pandas as pd
#sklearn


# AWS API endpoint
#API_ENDPOINT = "https://jy07on0qxh.execute-api.us-east-1.amazonaws.com/Test/predictphoneprice"
API_ENDPOINT = "https://prm6km3cwg.execute-api.us-east-2.amazonaws.com/Test/"
# Create your views here.
def index(request):
    result = None

    if request.method == "POST":
        #form = MobilePriceForm(request.POST)
        #if form.is_valid():
            # Prepare data for AWS API
        #    data = form.cleaned_data
        #    aws_endpoint = "https://your-aws-endpoint.amazonaws.com/predict"
        # Collect form data from the POST request

        # Convert boolean-like strings to integers
        blue = 1 if request.POST.get("blue", "false").lower() == "true" else 0
        dual_sim = 1 if request.POST.get("dual_sim", "false").lower() == "true" else 0
        four_g = 1 if request.POST.get("four_g", "false").lower() == "true" else 0
        three_g = 1 if request.POST.get("three_g", "false").lower() == "true" else 0
        touch_screen = 1 if request.POST.get("touch_screen", "false").lower() == "true" else 0
        wifi = 1 if request.POST.get("wifi", "false").lower() == "true" else 0

        battery_power = int(request.POST.get("battery_power", 0))
        clock_speed = float(request.POST.get("clock_speed", 0.0))
        front_camera = int(request.POST.get("fc", 0))
        internal_memory = int(request.POST.get("int_memory", 0))
        mobile_depth = float(request.POST.get("mobile_depth", 0.0))
        mobile_weight = int(request.POST.get("weight", 0))
        cores = int(request.POST.get("n_cores", 0))
        primary_camera = int(request.POST.get("pc", 0))
        pixel_height = int(request.POST.get("px_height", 0))
        pixel_width = int(request.POST.get("px_width", 0))
        ram = int(request.POST.get("ram", 0))
        screen_height = int(request.POST.get("sc_h", 0))
        screen_width = int(request.POST.get("sc_w", 0))
        battery_life = int(request.POST.get("battery_life", 0))

        phone_data_dict = [
            [
                battery_power, blue, clock_speed, dual_sim, front_camera,
                four_g, internal_memory, mobile_depth, mobile_weight, cores, primary_camera,
                pixel_height, pixel_width, ram, screen_height,
                screen_width, battery_life, three_g, touch_screen, wifi
            ]
        ]

        # Load the scaler
        scaler = joblib.load("scaler.pkl")

        # Full raw input (20 features)
        raw_input = np.array(phone_data_dict)

        # Indices of the 14 features that were scaled during training
        scaled_feature_indices = [0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

        # Column names used during training
        feature_columns = [
            'battery_power', 'clock_speed', 'fc', 'int_memory', 'm_dep',
            'mobile_wt', 'n_cores', 'pc', 'px_height', 'px_width',
            'ram', 'sc_h', 'sc_w', 'talk_time'
        ]

        # Extract the 14 features for scaling as a DataFrame with consistent column names
        features_to_scale_df = pd.DataFrame(raw_input[:, scaled_feature_indices], columns=feature_columns)

        # Scale the extracted features
        scaled_features = scaler.transform(features_to_scale_df)

        # Replace the original 14 features with the scaled ones in the raw input
        raw_input[:, scaled_feature_indices] = scaled_features

        # Prepare the payload for the API
        formatted_input = ",".join(map(str, raw_input[0]))
        data = {"body": f"[[{formatted_input}]]".replace(" ", "")}

        # Make a POST request to the API
        #response = requests.post(url=API_ENDPOINT, json=data)

        # Print the response
        #print("Prediction:", response.text)





        #phone_data_dict = '[[859,0,6,1,11,1,19,4,139,5,16,952,1713,2473,10,4,17,1,1,1]]'
        flattened_list = str(phone_data_dict[0]).replace(" ", "")
        # Wrapping in double square brackets to match the format
        formatted_string = f"[{flattened_list}]"

        #data = {"body": formatted_string}

        headers = {'Content-Type': 'application/json'}


        try:
            response = requests.post(url=API_ENDPOINT, json=data, headers=headers)
            if response.status_code == 200:
                print(response.json())
                result = response.json()
            else:
                result = f"Error: {response.status_code}, {response.text}"
        except requests.exceptions.RequestException as e:
                result = f"Request failed: {str(e)}"

    if result == "Low Cost":
        result = "$50 - $150"
    elif result == "Medium Cost":
        result = "$150 - $400"
    elif result == "High Cost":
        result = "$400 - $700"
    elif result == "Very High Cost":
        result = "$700 and above"

    return render(request, "PricePrediction/index.html", {"result": result})