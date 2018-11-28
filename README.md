# Flask & Keras

Flask server running a XOR Keras model.

# Keras model

This very basic Keras model learns the XOR operation. Run the model using `python train.py`.

```python
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([[0],[1],[1],[0]])

model = Sequential()
model.add(Dense(8, input_dim=2, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.1))
model.fit(X, Y, batch_size=1, nb_epoch=1000)
model.save('xor_model')
```

# Flask server

Run the server using `python server.py`.

```python
from flask import Flask, request
from keras.models import load_model
import tensorflow as tf
import numpy as np
import flask

app = Flask(__name__)
graph = tf.get_default_graph()
model = load_model('xor_model')

@app.route('/predict')
def predict():
    a = request.args['a']
    b = request.args['b']
    with graph.as_default():
        result = model.predict(np.array([[a,b]]))[0].tolist()
        data = {'result': result}
        return flask.jsonify(data)

app.run(host='0.0.0.0', debug=False)
```

# Testing

You can make a GET request using your browser :

```
http://ip_address:5000/predict?a=0&b=1
```
