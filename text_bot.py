#!/usr/bin/env python

import cgi
import cgitb
cgitb.enable()

import tensorflow as tf
#
import numpy as np
#
from load_text_generator import *





def print_header():
    print ("""Content-type: text/html\n
    <!DOCTYPE html>
    <html>
    <body>""")

def print_close():
    print ("""</body>
    </html>""")


def display_data(output):
    print_header()
    print('<p>'+output+'<\p>')
    print_close()

def main():

    form = cgi.FieldStorage()

    #N_characters = int(form["nchar"].value)
    #Randomness = float(form["nrand"].value)
    #seed = form["seed"].value
	
	N_characters=1000
	Randomness=0.4
	seed=['UFOs']


    display_data(seed)
     
   
    Initialize(one_step_model, seed)    
    
    output = Generate(one_step_model, N_characters, seed)
    display_data(output)

    f = open('out.log','a')
    f.write(output)
    f.close()


main()
