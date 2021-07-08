#!/usr/bin/env python

#import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from load_text_generator import *


from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/')
def index():
  return render_template('index.html', seed_initial="CARLSON:")

@app.route('/out/', methods = ['GET', 'POST'])
def my_link():

  req = request.form
  seed = [req['seed']]
  N_characters = int(req['nchar'])
 
  output = Generate(one_step_model, N_characters, seed) 
  
  print(output)
  print()
  print('seed', seed, seed[0])
  print()
  return render_template('index.html', output=output, seed_initial=seed[0])


if __name__ == '__main__':
  app.run(debug=True)
