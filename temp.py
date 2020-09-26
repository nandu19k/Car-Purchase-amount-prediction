from tensorflow.keras.models import load_model
from flask import Flask , request ,url_for ,flash, render_template
import numpy as np
import pickle


sc = pickle.load(open('sc.pkl' , 'rb'))

model = load_model('ann.h5' , compile=False)
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/amount' , methods=['GET' , 'POST'])
def purchase_amount():
    inputs = [int(x) for x in request.form.values()]
    inputs = np.array([inputs])
    inputs = sc.transform(inputs)
    output = model.predict(inputs)
    outputs = sc.inverse_transform(output)
    return render_template('home.html' , y = outputs)

if __name__ =='__main__':
    app.run(debug=True)
