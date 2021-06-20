from django.http.response import HttpResponseRedirect
from django.shortcuts import render
from django.http import HttpResponseRedirect
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
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

def prepare(text):
    test_sentence = [text]
    tokenizer.fit_on_texts(test_sentence)
    test_sequences =  tokenizer.texts_to_sequences(test_sentence)
    test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    return test_data

def cognitiveView(request):
    #pass the result of the prediction to the html page in a json
    #prediction = "Overgeneralization"
    if request.session.has_key('prediction'):
        prediction = request.session.get('prediction')
        del request.session['prediction']

    return render(request, 'cognitive.html', locals())

def addsentence(request):
    s = request.POST['content']
    prepared_text = prepare(s)  
    result = np.argmax(model.predict(prepared_text), axis=-1)
    category = Categories[result[0]]
    #use the model to predict
    #somehow pass it to cognitiveView as prediction
    #redirect the browser to '/cognitive/'
    request.session['prediction'] = category
    return HttpResponseRedirect('/cognitive/')
