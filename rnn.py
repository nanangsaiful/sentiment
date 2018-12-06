# -*- codi ng: utf-8 -*-
"""
Created on Tue Jun 05 18:26:43 2018

@author: nanang saiful
"""
import json
from keras import preprocessing
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
import pandas as pd
import re
import numpy
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import train_test_split
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import pickle 

df = pd.read_csv("C:/Users/nanang saiful/Downloads/Compressed/tugas/crnn-master3/datasentiment.csv",                  sep=",")
print ("preposesing")
#lower
print("olower")
df['text']=df['text'].apply(lambda x: x.lower())
#menghilangkan http
print("http")
df['text']=df['text'].apply(lambda x: re.sub(r"http\S+", '', x))
#mengambil huruf saja
df['text']=df['text'].apply(lambda x: re.sub('[^a-zA-z\s]','',x))
df['text']=df['text'].apply(lambda x: re.sub('rt','',x))

#stemming
#print("stem")
#stemfactory = StemmerFactory()
#stemmer = stemfactory.create_stemmer()
#df['text']=df['text'].apply(lambda x: stemmer.stem(x))
###stopword removal
#stopwfactory = StopWordRemoverFactory()
#stopwords = stopwfactory.create_stop_word_remover()
#df['text']=df['text'].apply(lambda x: stopwords.remove(x))
##menghilangkan rt

#max_fatures = 500
#tokenizer = Tokenizer(num_words=max_fatures, split=' ')

Y = pd.get_dummies(df['sentiment']).values
X_train, X_test, Y_train, Y_test = train_test_split(df['text'],Y, test_size = 0.33, random_state = 42)

tokenizer = Tokenizer( split=' ')
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test=tokenizer.texts_to_sequences(X_test)
#X_train = preprocessing.sequence.pad_sequences(X_train)
#X_test = preprocessing.sequence.pad_sequences(X_test,X_train.shape[1])
#
#print ("save tokenizer")
#with open('tokenizerstemstopword.pickle', 'wb') as handle:
#    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
#print("build model")
##inisialisasi parameter
#embed_dim = 128
#lstm_out = 196
##
#model = Sequential()
##mebuat layer embbeding dengan hasil matrik 1*input*embed_dim
#model.add(Embedding(len(tokenizer.word_index)+1, embed_dim,input_length = X_train.shape[1]))
#model.add(SpatialDropout1D(0.4))
##laye untuk rnn menggunakan lstm dengan output sejumlah lstm_out
#model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
##layer output berjumlah 3 neuron
#model.add(Dense(3,activation='softmax'))
#
#model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
#print(model.summary())
#
#print(X_train.shape,Y_train.shape)
#print(X_test.shape,Y_test.shape)
#batch_size = 32
##proses training
#model.fit(X_train, Y_train, epochs = 10, batch_size=batch_size, verbose = 2)
#
#print("save model")
## save model ke  JSON
#model_json = model.to_json()
#with open("modelsgdwithNOstemstopword.json", "w") as json_file:
#    json_file.write(model_json)
#    
## save model
#model.save("modelsgdwithNOstemstopwordSD.h5")
#print("Saved model to disk")
#
##proses prediksi
#predik=pd.get_dummies(model.predict_classes(X_test))
#report=classification_report(Y_test,predik)
#acc=accuracy_score(Y_test,predik)
#print("acc : "+str(acc))
#print("report : \n"+report)
