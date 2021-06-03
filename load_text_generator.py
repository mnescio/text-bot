#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf
# tf version 1.15.0

import numpy as np

one_step_model = tf.saved_model.load('test_model')
    
    
def Generate(one_step_model, N_characters, seed):
    states = None

    #next_char = hub.tf_utils.tf.constant(seed)
    next_char = tf.constant(seed)


    result = [next_char]
    for n in range(N_characters):
      next_char, states = one_step_model.generate_one_step(next_char, states=states)
      result.append(next_char)


    string = ''
    for c in np.array(result):
        string = string + str(c[0])[2:-1]
    
    return string







