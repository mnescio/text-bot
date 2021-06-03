#!/usr/bin/env python
# coding: utf-8


#import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

one_step_model = hub.load('test_model')


    
    
def Generate(one_step_model, N_characters, seed):
    states = None

    next_char = hub.tf_utils.tf.constant(seed)

    result = [next_char]
    for n in range(N_characters):
      next_char, states = one_step_model.generate_one_step(next_char, states=states)
      result.append(next_char)


    output = hub.tf_utils.tf.strings.join(result)[0].numpy().decode("utf-8")

    print(output)
    return output







