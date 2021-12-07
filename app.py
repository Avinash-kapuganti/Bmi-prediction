from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__,template_folder='template')

model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("BMI_predict.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    prediction=model.predict(final)
    print(prediction)
    #output='{0:.{1}f}'.format(prediction[0][2], 5)
    
    l=["Extremely weak","Weak","Normal","Overweight","Obesity","Extreme obesity"] 
    for i in prediction:
        ans= l[i]
    return render_template('BMI_predict.html',pred='Your {}'.format(prediction))
   


if __name__ == '__main__':
    app.run(debug=True)
    