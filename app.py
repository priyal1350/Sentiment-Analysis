from flask import Flask,render_template,request
from model import sentiment_predictor

import pickle

model=pickle.load(open('models.pkl','rb'))
app=Flask(__name__) #flask object

@app.route('/') #default route
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict_sentiment():
    text=request.form.get('textt')

    #predict
    result=sentiment_predictor([text])
    if result==0:
      result='anger'
    elif result==1:
        result='fear'
    elif result==2:
        result='joy'
    elif result==3:
        result='love'
    elif result==4:
        result='sadness'
    else:
        result='surprise'
    return render_template('index.html',result=result)

if __name__ == '__main__':
    app.run(debug=True)
