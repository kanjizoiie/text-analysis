import nltk
from nltk import corpus

tokens = nltk.word_tokenize("""
Parts of speech tagging can be important for syntactic and semantic analysis.
So, for something like the sentence above the word can has several semantic meanings.
One being a modal for question formation, another being a container for holding food or liquid, and yet another being a verb denoting the ability to do something.
Giving a word such as this a specific meaning allows for the program to handle it in the correct manner in both semantic and syntactic analyses.
""")

print("Parts of Speech", nltk.pos_tag(tokens))