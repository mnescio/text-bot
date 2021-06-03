#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf
import numpy as np

import cgi
import cgitb
cgitb.enable()


one_step_model = tf.saved_model.load('test_model')



# ## Generate Sample Text

def Initialize(one_step_model, seed):
    
    
    seed5=seed*5


    states = None

    next_char = tf.constant(seed)
    result = [next_char]

    for n in range(1000):
      next_char, states = one_step_model.generate_one_step(next_char, states=states)
      result.append(next_char)

    result = tf.strings.join(result)



    states = None

    next_char = tf.constant(seed5)

    result = [next_char]

    for n in range(1000):
      next_char, states = one_step_model.generate_one_step(next_char, states=states)
      result.append(next_char)

    result = tf.strings.join(result)
    
    
    
def Generate(one_step_model, N_characters, seed):
    states = None

    next_char = tf.constant(seed)
    result = [next_char]
    for n in range(N_characters):
      next_char, states = one_step_model.generate_one_step(next_char, states=states)
      result.append(next_char)


    output = tf.strings.join(result)[0].numpy().decode("utf-8")
    print('\n'+output+'\n')
    return output







