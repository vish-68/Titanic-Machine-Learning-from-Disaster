# -*- coding: utf-8 -*-
"""
Created on Thr Sep 23 15:14:04 2021

@author: VISHAL UPARE
"""
# %%

import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

# %%

app = Flask(__name__)

# %%

model = pickle.load(open("knn_tune.pkl", "rb"))

# %%
pclass={1:"I Class",2:"II Class",3:"III Class"}

sex={0:"Female", 1:"Male"}

embarked={0:"Cherbourg", 1:"Queenstown", 2:"Southampton"}

#%%

@app.route("/")
def home():
    return render_template("home.html")

# %%


@app.route("/TMLD", methods=['POST'])
def TMLD():
    data = request.form["names"]
    data1 = request.form["Pclass"]
    data2 = request.form["Sex"]
    data3 = request.form["Age"]
    data4 = request.form["SibSp"]
    data5 = request.form["Parch"]
    data6 = request.form["Fare"]
    data7 = request.form["Embarked"]

    arr = np.array([[data1, data2, data3, data4, data5, data6, data7]])
    prediction = model.predict(arr)

    # create original output dict
    output_dict1 = dict()
    output_dict1['Name'] = data
    output_dict1['Passenger Class'] = pclass[int(data1)]
    output_dict1['Sex'] = sex[int(data2)]
    output_dict1['Age'] = data3
    output_dict1['Number of Siblings / Spouses aboard'] = data4
    output_dict1['Number of Parents / Children aboard'] = data5
    output_dict1['Fare'] = data6
    output_dict1['Port of Embarked'] = embarked[int(data7)]

    
    return render_template("result.html", original_input1=output_dict1, data=prediction)
# %%


if __name__ == '__main__':
    app.run(debug=True)

# %%
