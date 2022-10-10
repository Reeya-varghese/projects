import numpy as np
import pickle
import joblib
import matplotlib
import matplotlib.pyplot as plt
import time
import pandas
import os
from flask import Flask,request,jsonify,render_template

app = Flask(__name__)

model=pickle.load(open('C:/Users/reeya/PT1/customer_segmentation/cust_xgbmodel.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/plat')
def plat():
    return  render_template('plat.html') 

@app.route('/dia')
def dia():
    return  render_template('dia.html')  

@app.route('/advertise')
def advertise():
    return  render_template('advertisement.html') 

@app.route('/gold')
def gold():
    return  render_template('gold.html') 

@app.route('/silver')
def silver():
    return  render_template('silver.html') 

@app.route('/delete')
def delete():
    return  render_template('delete.html')  

@app.route('/hpc')
def hpc():
    return  render_template('hpc.html')  

@app.route('/pc')
def pc():
    return  render_template('pc.html')  

@app.route('/npc')
def npc():
    return  render_template('npc.html')  

@app.route('/registered')
def registered():
    return  render_template('registered.html')  

@app.route('/predict',methods=["POST","GET"])
def predict():
    input_feature=[float(x) for x in request.form.values()]
    features_values=[np.array(input_feature)] 
    names=[['Sex','Marital status','Age','Education','Income','Occupation','Settlement size']]

    data=pandas.DataFrame(features_values,columns=names)
    prediction=model.predict(data)
    print(prediction)

    if(prediction== 0):
        return render_template("npc.html",prediction_text="Not a potential customer")   
    elif(prediction== 1):
        return render_template("pc.html",prediction_text="Potential customer")     
    else:
        return render_template("hpc.html",prediction_text="Highly potential customer")    

if __name__=="__main__":
    port=int(os.environ.get('POST',5000))
    app.run(port=port,debug=True,use_reloader=False)