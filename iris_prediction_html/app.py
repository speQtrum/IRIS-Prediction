from flask import Flask, render_template,flash, session, redirect, url_for, session
from flask_wtf import FlaskForm
from wtforms import FloatField,SubmitField
from wtforms.validators import DataRequired
import numpy as np
import pickle


app = Flask(__name__)
model = pickle.load(open('/home/ubuntu/flaskapp/webapp_102/model.pkl', 'rb'))
app.config['SECRET_KEY'] = 'mysecretkey'




class Iris_Form(FlaskForm):
    sepal_length = FloatField('Sepal Length',validators=[DataRequired()])
    sepal_width = FloatField('Sepal Width ',validators=[DataRequired()])
    petal_length  = FloatField('Petal Length',validators=[DataRequired()])
    petal_width = FloatField('Petal Width ',validators=[DataRequired()])
    submit = SubmitField('Submit')




@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = False
    form = Iris_Form()
    if form.validate_on_submit():

        sepal_length = form.sepal_length.data
        sepal_width = form.sepal_width.data
        petal_length = form.petal_length.data
        petal_width = form.petal_width.data  


        input_features = [sepal_length,sepal_width,petal_length,petal_width]
        final_features = [np.array(input_features)]
        prediction = model.predict(final_features)[0]

    return render_template('index.html', form = form, prediction = prediction)



if __name__ == '__main__':
    app.run(host='0.0.0.0',port=80,debug=True)
