import numpy as np
from flask import Flask, request,render_template
from joblib import load


app = Flask(__name__)
model = load('xgb.joblib') 

@app.route('/')
def home():
    return render_template('./index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    result = "অত্যন্ত দুঃখিত, আপনি সারভিকাল ক্যান্সারে আক্রান্ত হওয়ার ঝুঁকিতে রয়েছেন। দেরি না করে আমাদের বিশেষজ্ঞ ডাক্তারদের সাথে আজই যোগাযোগ করুন।"
    int_features = [x for x in request.form.values()]
    # final_features = np.array(int_features, dtype=float)
    int_features = np.array([int_features], dtype=int)
    # int_features = int_features[0].reshape(1, -1)
    # print(int_features[0])
    prediction = model.predict(int_features)
    print(int_features)
    print(prediction)
    if(prediction[0]==0):
        result = "অভিনন্দন, আপনি সারভিকাল ক্যান্সারে আক্রান্ত হওয়ার ঝুঁকিতে নেই।"
    return render_template('./index.html', prediction_text='{}'.format(result))

if __name__ == "__main__":
    app.run(debug=True)