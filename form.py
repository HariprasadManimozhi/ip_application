from flask_wtf import FlaskForm
from wtforms import TextField, IntegerField, SubmitField


class CreateTask(FlaskForm):
    emp_id = TextField('Employee ID')
    emp_name = TextField('Employee Name')
    branch = TextField('Branch')
    create = SubmitField('Create')
    snap = SubmitField('Snap')
