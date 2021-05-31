# Text Bot

## the text_bot.ipynb notebook loads a previously trained text generator

Text generator adapted from https://www.tensorflow.org/text/tutorials/text_generation

Transcripts taken from Tucker Carlson Tonight episodes:

```
February 26, 2021
April 5, 2021
April 29, 2021
April 30, 2021
May 1, 2021
May 13, 2021
May 20, 2021
```

and saved in file ```transcripts_all.txt```

Replace this with any plain text file to customize the text generator.


To run the text bot, enter in your terminal:

```
git clone https://github.com/mnescio/text-bot.git
cd text-bot
jupyter notebook --ip 0.0.0.0 --no-browser
```

Run the text_bot.ipynb notebook

Jupyter notebooks: https://jupyter.org/install.html



# Run in Google Colab

If you want to run this on google colab instead of downloading all the python modules needed to your own machine, follow these instructions:

1. Go to https://colab.research.google.com/
2. Click "NEW NOTEBOOK" in the bottom right corner
3. Enter the code below

```

from google.colab import drive
drive.mount('/content/gdrive/')

# # you will be asked to click on a link to copy a code to log into your google drive

%mkdir gdrive/MyDrive/text-bot

%cd gdrive/MyDrive/text-bot

! git clone https://github.com/mnescio/text-bot.git

%cd text-bot/

# this will show you the files you have downloaded into your google drive
! ls 

! jupyter nbconvert --to python text_bot.ipynb

# this line of code will generate the output. You can copy and paste just the following line
# to generate new lines of output. To change the seed phrase, you will need to edit
# the notebook text_bot.ipynb and rerun the line above (! jupyter nbconvert --to python text_bot.ipynb)

! python text_bot.py

```

You will have to edit the text_bot.ipynb notebook in your google drive, where it will be saved, and then run 

```
! jupyter nbconvert --to python text_bot.ipynb
! python text_bot.py
```
to generate new output.

Alternatively you can just repeatedly run the line

```
! python text_bot.py
```

to generate output with the default seed phrase


