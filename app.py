import flask
import pickle
import numpy as np
from markupsafe import escape
app = flask.Flask(__name__, template_folder='templates')
model = pickle.load(open('model/knn_model.pkl', 'rb'))
model = model['knn_']


@app.route('/')
def index():
    return flask.render_template('main.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in flask.request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = {0: 'not placed', 1: 'placed'}

    return flask.render_template('main.html', prediction_text='Student must be {} to workplace'.format(output[prediction[0]]))


@app.route('/hello')
def hello_world():
    return 'Hello World'


@app.route('/hello/<username>')
def sayHello(username):
    return 'Hello %s ' % escape(username)


if __name__ == "__main__":
    app.run(debug=True)
