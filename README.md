# Text Bot

Web App: https://tuckered-out-bot.herokuapp.com/

## the text_bot.ipynb notebook loads a previously trained text generator

Text generator adapted from https://www.tensorflow.org/text/tutorials/text_generation

Transcripts taken from Tucker Carlson (who is evil, just like this bot) Tonight episodes:

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


To run the text bot, enter in your linux terminal (how to find for [windows](https://docs.microsoft.com/en-us/windows/wsl/install-win10), [mac](https://www.howtogeek.com/682770/how-to-open-the-terminal-on-a-mac/)):

```
git clone https://github.com/mnescio/text-bot.git
cd text-bot
bash bootstrap.sh
jupyter notebook --ip 0.0.0.0 --no-browser
```

Run the text_bot.ipynb notebook

Jupyter notebooks: https://jupyter.org/install.html


