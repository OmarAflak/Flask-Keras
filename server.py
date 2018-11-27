from flask import Flask, request
from keras.models import load_model
import tensorflow as tf
import numpy as np
import flask

app = Flask(__name__)
graph = tf.get_default_graph()
model = load_model('xor_model')

@app.route('/')
def hello():
    data = {'result': 'Hello World!'}
    return flask.jsonify(data)

@app.route('/predict')
def predict():
    input = request.args.getlist('x', type=float)
    with graph.as_default():
        result = model.predict(np.array([input]))[0].tolist()
        data = {'result': result}
        return flask.jsonify(data)

app.run(host='0.0.0.0', debug=False)
