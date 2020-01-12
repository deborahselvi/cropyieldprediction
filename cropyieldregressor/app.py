from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import os
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

cur_dir = os.path.dirname(__file__)
# cur_dir = 'https://www.pythonanywhere.com/user/deborahselvi/files/home/deborahselvi/mysite'
# clf = pickle.load(open(os.path.join(cur_dir,'pkl_objects','crop_yield_dt.pkl'),'rb'))

def predict_yield(state,district,season,crop,area):
    url = "apy.csv"
    # url = os.path.join(cur_dir,'mysite','apy.csv')

    crop_prod= pd.read_csv(url)
    
    crop_prod = crop_prod.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    
    crop_prod['Area'] = pd.to_numeric(crop_prod['Area'], errors='coerce')
    crop_prod['Production'] = pd.to_numeric(crop_prod['Production'], errors='coerce')
    
    crop_prod_dummies = pd.get_dummies(crop_prod)
    print(range(crop_prod_dummies.shape[1]))
    
    crop_prod_dummies = crop_prod_dummies.drop(columns=['Crop_Year'])
    
    features = crop_prod_dummies.loc[:, crop_prod_dummies.columns != 'Production']
    
    cur_dir = os.getcwd()
    tree_pickle = pickle.load(open(os.path.join(cur_dir,
                                                # 'pkl_objects',
                                                'crop_yield_dt.pkl'),'rb'))
    # tree_pickle = pickle.load(open(os.path.join(cur_dir,'mysite','pkl_objects','crop_yield_dt.pkl'),'rb'))
    print(range(features.shape[1]))
    index_dict = dict(zip(features.columns,range(features.shape[1])))
    print(index_dict)
    new_vector = np.zeros(810)
    try:
        new_vector[index_dict['State_Name_'+state]] = 1
    except:
        pass
    try:
        new_vector[index_dict['District_Name_'+district]] = 1
    except:
        pass
    try:
        new_vector[index_dict['Season_'+season]] = 1
    except:
        pass
    try:
        new_vector[index_dict['Crop_'+crop]] = 1
    except:
        pass
    try:
        new_vector[index_dict['Area']] = area
    except:
        pass
    
    print("new_vector {}".format(new_vector))
    y_pred = tree_pickle.predict([new_vector]) 
      
    # print the predicted yield 
    print("Predicted yield: % d\n"% y_pred)  
    return y_pred
    
# def train(document, y):
#     X = document
#     clf.partial_fit(X, [y])

# class CropInfoForm(Form):
#     CropInfo = TextAreaField('',[validators.DataRequired()])
    
@app.route('/')
def index():
    form = request.form
    return render_template('crop_yield_app.html', form=form)

@app.route('/results', methods=['POST'])
def results():
    form = request.form
    if request.method == 'POST':
        state = request.form['state']
        district = request.form['district']
        season = request.form['season']
        crop = request.form['crop']
        area = request.form['area']
        y = predict_yield(state,district,season,crop,area)
        return render_template('show_resulf.html', prediction=y)
    return render_template('crop_yield_app.html',form=form)

if __name__ == '__main__':
    app.run()