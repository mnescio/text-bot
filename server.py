#!/usr/bin/env python

import tensorflow as tf
import numpy as np
from load_text_generator import *


from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/my-link/', methods = ['GET', 'POST'])
def my_link():

  req = request.form
  seed = req['seed']
  nchar = int(req['nchar'])
  nrand = float(req['nrand'])/10
  

  


  output = Generate(one_step_model, N_characters, seed) 



  return output

if __name__ == '__main__':
  app.run(debug=True)
