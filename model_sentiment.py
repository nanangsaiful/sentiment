# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 13:53:43 2018

@author: nanang saiful
"""
from keras.models import load_model
import pickle
from keras import preprocessing
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


#lakukan load model terlebih dahulu sebelum melakukan sentimen

def loadmodel(tokenfile,modelfile):
    global tokenizer
    global model
    global factory
    factory = StemmerFactory()
    global stemmer 
    stemmer = factory.create_stemmer()
    #load token file
    #token file ='tokenizer.pickle'
    with open(tokenfile, 'rb') as handle:
        tokenizer = pickle.load(handle)
    #load model rnn
    #modelfile = "model.h5"
    model = load_model(modelfile)
    print("Loaded model from disk")
    print (tokenizer,model)

def sentiment(kalimat):
    label=['negative', 'neutral', 'positive']
    #preprocessing
    #lower
    kalimat=kalimat.lower()
    #hapus alamat url
    kalimat=re.sub(r"http\S+", '', kalimat)
    #mengambil huruf saja
    kalimat=re.sub('[^a-zA-z\s]','',kalimat)
    #stemming
    kalimat=stemmer.stem(kalimat)
    #mengubagh kalimat menjadi squence
    X=tokenizer.texts_to_sequences([kalimat])
    #menyamakan ukuran kalimat dengan ukuran input model
    X = preprocessing.sequence.pad_sequences(X,maxlen=model.input_shape[1])
    #melakukan deteksi sentimwen
    hsl=model.predict(X)
    print (hsl)
    print ("dengan label : "+label[hsl.argmax()])

