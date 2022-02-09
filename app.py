# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from flask import Flask , request , render_template
import joblib


app = Flask(__name__)
model =  joblib.load("student_mark_predictor_model.pkl")

df = pd.DataFrame()

@app.route("/")
def home():
    print("In home")
    return render_template('index.html')

@app.route("/predict",methods=['POST'])
def predict():
    print("hi in predict")
    global df
    input_features = [int(x) for x in request.form.values()]
    features_value = np.array(input_features)

    #validate input hours
    if input_features[0] <0 or input_features[0] >24:
        return render_template('index.html',prediction_text='Please provide valid input between 1 io 24')

    output = model.predict([features_value])[0][0].round(2)

    if output >100:
        output = 100

    df = pd.concat([df,pd.DataFrame({'Study Hours':input_features,'Predicate Output':[output]})] , ignore_index=True)
    print(df)
    # store data for further 
    df.to_csv('smp_data_from_app.csv')
    
    return render_template('index.html' , prediction_text = 'you wil get [{}%] marks when you will study [{}] hours per day'.format(output, int(features_value[0])) )


if __name__ == '__main__':
    app.run()