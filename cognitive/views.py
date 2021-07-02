from json.encoder import JSONEncoder
from django.http.response import HttpResponseRedirect
from django.shortcuts import render
from django.http import HttpResponseRedirect
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import re
import json
# Create your views here.

df = pd.read_csv('./allnumbered.csv', names=['sentence', 'label', 'source'])
MAX_NB_WORDS = 10000
MAX_SEQUENCE_LENGTH = 100
tokenizer  = Tokenizer(num_words = MAX_NB_WORDS)
tokenizer.fit_on_texts(df['sentence'])
sequences =  tokenizer.texts_to_sequences(df['sentence'])
word_index = tokenizer.word_index
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

model = tf.keras.models.load_model("./LSTM2")
Categories = ["", "nondistorted", "Overgeneralization", "Should statement"]
Colors = ["", "", "Bright baige", "Bright pink"]

def prepare(text):
    test_sentence = [text]
    #tokenizer.fit_on_texts(test_sentence)
    test_sequences =  tokenizer.texts_to_sequences(test_sentence)
    test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    return test_data

def cognitiveView(request):
    categories = []
    colored = []
    if request.POST.get('content'):
        s = request.POST.get('content')
        listS = re.split(r'(?<=\w\.)\s', s)
        #loop on lists, prepare in and predict in the loop, and compile the answers
        for sentence in listS:
            prepared_text = prepare(sentence)
            result = np.argmax(model.predict(prepared_text), axis=-1)
            categories.append(Categories[result[0]])
            # coloredobj = {"sentence": sentence, "color": Colors[result[0]]}
            # coloredjson = json.dumps(coloredobj)
            # print(coloredjson)
            # colored.append(coloredjson)
            comma = re.sub(',', ' ', str(sentence))
            colored.append(Colors[result[0]])
            colored.append(comma)
    return render(request, 'cognitive.html', {'predictions': categories, 'colored': json.dumps(colored)})

