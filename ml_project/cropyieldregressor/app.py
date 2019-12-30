from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import os
import pickle
import numpy as np

app = Flask(__name__)

cur_dir = os.path.dirname(__file__)
clf = pickle.load(open(os.path.join(cur_dir,'pkl_objects','crop_yield_dt.pkl'),'rb'))

def classify(document):
    X = document
    y = clf.predict(X)[0]
    proba = np.max(clf.predict_proba(X))
    return proba
    
def train(document, y):
    X = document
    clf.partial_fit(X, [y])

class CropInfoForm(Form):
    CropInfo = TextAreaField('',[validators.DataRequired()])
    
@app.route('/')
def index():
    form = CropInfoForm(request.form)
    return render_template('crop_yield_app.html', form=form)

@app.route('/results', methods=['POST'])
def results():
    form = CropInfoForm(request.form)
    if request.method == 'POST' and form.validate():
        state = request.form['CropInfo']
        y, proba = classify(state)
        return render_template('show_resulf.html',info=state, prediction=y, probability=round(proba*100,2))
    return render_template('crop_yield_app.html',form=form)

if __name__ == '__main__':
    app.run()