import numpy as np
import pickle
import joblib
import matplotlib
import matplotlib.pyplot as plt
import time
import h5py
import librosa
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import os
import random

from base64 import b64decode
import pandas
from flask import Flask,request,jsonify,render_template

app = Flask(__name__)
config = tf.compat.v1.ConfigProto(
                intra_op_parallelism_threads=1,
                allow_soft_placement=True
            )
session = tf.compat.v1.Session(config=config)

#files=os.listdir(os.curdir)
#print(files)
tf.compat.v1.keras.backend.set_session(session)

model=load_model('./CNN-Model.h5', compile =False)
model.make_predict_function()

e=OneHotEncoder()
with open('encoder.pickle', 'rb') as f:
    e= pickle.load(f)

@app.route('/Prediction')
def Prediction():
    return  render_template('Prediction.html') 

@app.route('/')
def index_view():
   return render_template('index.html')

def extract_features_and_predict(path):
    filepath='tmp' + path 
    features=np.empty((0,162))
    x,sr =librosa.load(filepath)
    zcr = np.mean(librosa.feature.zero_crossing_rate(x).T,axis=0)

    stft=np.abs(librosa.stft(x))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T,axis=0)
   
    mfcc = np.mean(librosa.feature.mfcc(y=x, sr=sr).T, axis=0)
    
    rmse = np.mean(librosa.feature.rms(y=x).T, axis=0)

    mel = np.mean(librosa.feature.melspectrogram(y=x,sr=sr).T, axis=0)
    ext_features=np.hstack([zcr,chroma_stft,mfcc,rmse,mel])
    features=np.vstack([features,ext_features])
    x=features
    x=x.reshape(x.shape[1],x.shape[0])
    res_x=x
    preds = model_predict(res_x, model)

    os.remove(filepath)
    return preds


def model_predict(vector, model):
    try:
        with session.as_default():
                #with session.graph.as_default():
                    preds = model.predict(np.array([vector]))
                    
                    pred= e.inverse_transform(preds)
                    return pred
                    
    except Exception as ex:
        print(ex)    


@app.route('/predict',methods=["POST","GET"])
def predict():
    if request.method == "POST":
        f = request.files['audio_data']
        random_number = random.randint(00000, 99999)
        filepath='./tmp' +str(random_number)+ '.wav'
        f.save(filepath)
        print('file uploaded successfully')
            
        filename=str(random_number) +'.wav'
        res = extract_features_and_predict(filename)    
    
    if(res=='Calm'):
            return render_template("Prediction.html", prediction_text="The predicted emotion of the audio is : CALM ")   
    elif(res=='Neutral'):
            return render_template("Prediction.html", prediction_text="The predicted emotion of the audio is : NEUTRAL ")  
    elif(res=='Angry'):
            return render_template("Prediction.html", prediction_text="The predicted emotion of the audio is : ANGRY ")  
    elif(res=='Happy'):
            return render_template("Prediction.html", prediction_text="The predicted emotion of the audio is : HAPPY ")   
    elif(res=='Sad'):
            return render_template("Prediction.html", prediction_text="The predicted emotion of the audio is : SAD ")
    elif(res=='Disgust'):
            return render_template("Prediction.html", prediction_text="The predicted emotion of the audio is : DISGUST ")     
    elif(res=='Surprise'):
            return render_template("Prediction.html", prediction_text="The predicted emotion of the audio is : SURPRISE ")   
    elif(res=='Fear'):
            return render_template("Prediction.html", prediction_text="The predicted emotion of the audio is : FEAR ")           
    else:   
            return render_template("index.html")


if __name__ == "__main__":
     port=int(os.environ.get('POST',5000))
     app.run(port=port,debug=True,use_reloader=False)