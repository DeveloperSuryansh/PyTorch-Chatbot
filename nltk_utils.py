import nltk
import numpy as np
from nltk.stem import SnowballStemmer
from torchtext.data import get_tokenizer

nltk.download('nonbreaking_prefixes')
nltk.download('punkt')
tokenizer = get_tokenizer("toktok", language='en')
stemmer = SnowballStemmer('english')

def tokenize(sentence):
    return tokenizer(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):

    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    #print(tokenized_sentence)
    bag = np.zeros_like(all_words, dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0

    return bag

