# Text Bot

Web App: https://tuckered-out-bot.herokuapp.com/

[web app source code](https://github.com/mnescio/text-bot/tree/app)

## Instructions for running the code locally

Text generator adapted from https://www.tensorflow.org/text/tutorials/text_generation

The text_bot.ipynb notebook loads a previously trained model.

The text_generator.ipynb trains a model on the transcripts.

Transcripts taken from 35 Tucker Carlson (who is evil, just like this bot) Tonight episodes and saved in file ```transcripts.txt```


To run the text bot, enter in your linux terminal (how to find for [windows](https://docs.microsoft.com/en-us/windows/wsl/install-win10), [mac](https://www.howtogeek.com/682770/how-to-open-the-terminal-on-a-mac/)):

```
git clone https://github.com/mnescio/text-bot.git
cd text-bot
bash bootstrap.sh
jupyter notebook --ip 0.0.0.0 --no-browser
```

Run the text_bot.ipynb notebook

Jupyter notebooks: https://jupyter.org/install.html


