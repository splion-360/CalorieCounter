from flask import Flask,render_template,session,url_for,redirect
from flask_wtf import FlaskForm
from wtforms import TextField,SubmitField
import numpy as np
import pickle
def calorie_predict(model,sample_json):
    male = sample_json['male']
    female = sample_json['female']
    age = sample_json['age']
    weight = sample_json['weight']
    height = sample_json['height']
    duration = sample_json['duration']
    heartbeat = sample_json['heartbeat']
    body_temp = sample_json['body_temp']

    lis = np.array([male,female,age,weight,height,duration,heartbeat,body_temp]).reshape((1,8))
    return model.predict(lis)[0]

model = pickle.load(open('C:\\Users\\SuryaPrakash\\Downloads\\TF_2_Notebooks_and_Data\\09-Deployment\\final_model.pkl', 'rb'))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'
class CounterForm(FlaskForm):
    male = TextField('Male')
    female = TextField('Female')
    age = TextField('Age')
    weight = TextField('Weight')
    height = TextField('Height')
    duration = TextField('Duration')
    heartbeat = TextField('Heart BPM')
    body_temp = TextField('Body Temperature')

    submit = SubmitField('Analyze')

@app.route("/",methods = ["GET", "POST"])
def index():
    form = CounterForm()
    if form.validate_on_submit():
        session['male'] = form.male.data
        session['female'] = form.female.data
        session['age'] = form.age.data
        session['weight'] = form.weight.data
        session['height'] = form.height.data
        session['duration'] = form.duration.data
        session['heartbeat'] = form.heartbeat.data
        session['body_temp'] = form.body_temp.data

        return redirect(url_for("prediction"))
    return render_template('home.html',form=form)

@app.route('/prediction')
def prediction():
    content = {}
    content['male'] = float(session['male'])
    content['female'] = float(session['female'])
    content['age'] = float(session['age'])
    content['weight'] = float(session['weight'])
    content['height'] = float(session['height'])
    content['duration'] = float(session['duration'])
    content['heartbeat'] = float(session['heartbeat'])
    content['body_temp'] = float(session['body_temp'])
    results = calorie_predict(model,content)

    return render_template('prediction.html',results=results)
if __name__ == '__main__':
    app.run()