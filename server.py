from flask import Flask, request
from keras.models import load_model
import tensorflow as tf
import numpy as np
import flask

app = Flask(__name__)
model = load_model('xor_model')
graph = tf.get_default_graph()

# handle request (GET by default)
@app.route('/')
def hello():
    data = {'result': 'Hello World!'}
    return flask.jsonify(data)

# request model prediction
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        a = request.args['a']
        b = request.args['b']
    elif request.method == 'POST':
        a = request.form['a']
        b = request.form['b']

    # Required because of a bug in Keras when using tensorflow graph cross threads
    with graph.as_default():
        result = model.predict(np.array([[a,b]]))[0].tolist()
        data = {'result': result}
        return flask.jsonify(data)

# start Flask server
app.run(host='0.0.0.0', port=5000, debug=False)
