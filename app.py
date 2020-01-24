#libraries
from flask import Flask, request, redirect, url_for, render_template,Response,jsonify,session
#from emo import process_facedetection

import time
import os
from os import listdir
from os.path import isdir, join, isfile, splitext

from pymongo import MongoClient

from form import *
from testing_webcam_flask import *
from training import *
from agegenderemotion_webcam_flask import *
from testing_counting import *

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
print(APP_ROOT)

@app.route('/a1')
def a1():
    # get data from database or mqtt whatever...
    title = 'Hi'
    labels = [1,5,6,9]
    data = [10,20,30,40,50]

    return render_template('chartjs.html', data={'title': title, 'data': data, 'labels': labels} )

@app.route('/')
def index():
    data = [10,20,30,40,50]
    return render_template("index.html",data={'data': data })

@app.route('/a')
def ind():
    return Response(open('./templates/webcam.html').read(), mimetype="text/html")

@app.route('/image', methods=['POST'])
def image():
    i = request.files['image']  # get the image
    f = ('%s.jpeg' % time.strftime("%Y%m%d-%H%M%S"))
    i.save('%s/%s' % (APP_ROOT, f))

app.config.update(dict(SECRET_KEY='yoursecretkey'))
client = MongoClient('localhost:27017')
db = client.TaskMan

if db.settings.find({'name': 'task_id'}).count() <= 0:
    print("task_id Not found, creating....")
    db.settings.insert_one({'name':'task_id', 'value':0})

def updateTaskID(value):
    task_id = db.settings.find_one()['value']
    task_id += value
    db.settings.update_one(
        {'name':'task_id'},
        {'$set':
            {'value':task_id}
        })

def createTask(form):
    emp_id = form.emp_id.data
    emp_name = form.emp_name.data
    branch = form.branch.data
    task_id = db.settings.find_one()['value']
    
    task = {'id':task_id, 'emp_id':emp_id, 'branch':branch, 'emp_name':emp_name}

    db.tasks.insert_one(task)
    updateTaskID(1)
    return redirect('/b')


@app.route('/b', methods=['GET','POST'])
def main():
    # create form
    cform = CreateTask(prefix='cform')

    # response
    if cform.validate_on_submit() and cform.create.data:
        upload(cform)
        return createTask(cform)

    # read all data
    docs = db.tasks.find()
    data = []
    for i in docs:
        data.append(i)

    return render_template('webcam.html', cform=cform, data = data)


def upload(cform):
    target = os.path.join(APP_ROOT,"datasets/")
    if not os.path.isdir(target):
        os.mkdir(target)
    classfolder = str(request.form['cform-branch'])
    emp_id = str(request.form["cform-emp_id"])
    session['classfolder'] = classfolder
    target1 = os.path.join(target,str(request.form["cform-branch"])+"_"+str(request.form['cform-emp_id'])+"/")
    if not os.path.isdir(target1):
        os.mkdir(target1)
    session['target1']=target1
    print(target1)
    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = "/".join([target1,filename])
        print(destination)
        file.save(destination)
    return 'OK'

@app.route('/web')
def webcam():
    return render_template('webcamstream.html')

# Entry point for web app
@app.route('/video_viewer')
def video_viewer():
    return Response(cumucount(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Entry point for web app
@app.route('/video_viewer2')
def video_viewer_i():
    return Response(process_facerecognition(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Entry point for web app
@app.route('/video_viewer1')
def video_viewer_emo():
    return Response(process_facedetection1(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/w')
def foo():
    mainhp(parse_arguments(sys.argv[1:]))
    data = [10,20,30,40,50]
    return render_template('index.html',data={'data': data })

# Initialize for web app
#@app.route('/b')
#def index1():
#    return render_template('web_app_flask.html')

# Entry point for web app
#@app.route('/video_viewer')
#def video_viewer():
#    return Response(process_facedetection(), mimetype='multipart/x-mixed-replace; boundary=frame')
