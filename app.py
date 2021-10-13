import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_features = [int(x) for x in request.form.values()]
    final_features = [np.array(input_features)]
    prediction = model.predict(final_features)
    
    return render_template('index.html', prediction_text='The Predicted Median value of owner-occupied homes in $1000\'s is {}'.format(prediction[0]))

if __name__=="__main__":
    app.run(debug=True)