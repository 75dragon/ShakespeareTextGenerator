# keras module for building LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
import keras.utils as ku

import pickle

# set seeds for reproducability
# from tensorflow import set_random_seed
# from numpy.random import seed
# set_random_seed(2)
# seed(1)

import pandas as pd
import numpy as np
import string, os, sys
print(sys.path)
#conda create -n myenv python=3.6
#conda activate myenv
#conda deactivate

play_df = pd.read_csv("./Shakespeare_data.csv")

all_lines = [h for h in play_df.PlayerLine]

print(len(all_lines))

def cleanText(text):
    text = text.replace("-", " ")
    text = "".join(v for v in text if v not in string.punctuation).lower()
    text = text.encode("utf8").decode("ascii",'ignore')
    if (text[:3] == "act" or text[:5] == "scene"):
        return ""
    return text

corpus = [cleanText(x) for x in all_lines]
while "" in corpus: corpus.remove("")
print(corpus[:10])


tokenizer = Tokenizer()

#take the first 7k lines to train
#.fit_on_texts() tokenizes the text, 1 = most poplular word, 2 = 2ct
#.texts_to_sequences() changes those texts to an char sequence based of tokenize
#then for some reason, first pairs, then 3, 4... to end of sentice.

def get_sequence_of_tokens(corpus):
    ## tokenization
    corpus = corpus[:7000]
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1

    ## convert data to sequence of tokens
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    return input_sequences, total_words

inp_sequences, total_words = get_sequence_of_tokens(corpus)
inp_sequences[:10]

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

#now all arrays same length, filled with empty string first?
#input_sequences is an array of arrays of same length maxlen of seq, padded with 0's
#predictor is the array of arrays of words beforehard -> label is the shold be word
#to_categorical = Converts a class vector (integers) to binary class matrix.
def generate_padded_sequences(input_sequences):
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    #print(input_sequences)
    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    #print(predictors)
    #print(label)
    label = ku.to_categorical(label, num_classes=total_words)
    return predictors, label, max_sequence_len

predictors, label, max_sequence_len = generate_padded_sequences(inp_sequences)
predictors.shape, label.shape

#sequential model, build the model layer by layer
def create_model(max_sequence_len, total_words):
    input_len = max_sequence_len - 1
    model = Sequential()
    print(max_sequence_len)
    # Add Input Embedding Layer
    # this is the first row of circles
    # input 1 = total words, 2 = size of vector space that represents word, max_sequence_len is the longest sentense in shake
    model.add(Embedding(total_words, 10, input_length=input_len))

    # Add Hidden Layer 1 - LSTM Layer
    #adds lstm layer
    #dropout layer, also scales others larger
    model.add(LSTM(512))
    model.add(Dropout(0.4))

    # Add Output Layer
    #output space - softmax 0-1 all pos
    model.add(Dense(total_words, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model

model = create_model(max_sequence_len, total_words)
model.summary()

#epocs is repeats, verbose just shows progress
model.fit(predictors, label, epochs=2, verbose=1)

model.fit(predictors, label, epochs=20, verbose=2)
#
model.fit(predictors, label, epochs=20, verbose=0)
Pickled_LR_Model = None
Pkl_Filename = "Pickle_RL_Model.pkl"
with open(Pkl_Filename, 'wb') as file:
    pickle.dump(model, file)

with open(Pkl_Filename, 'rb') as file:
    Pickled_Model = pickle.load(file)

#def generate_text(seed_text, next_words, model, max_sequence_len):
def generate_text(seed_text, next_words, Pickled_Model, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)

        output_word = ""
        for word,index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " "+output_word
    return seed_text.title()

print ("1. ",generate_text("Julius", 20, model, max_sequence_len))
print ("2. ",generate_text("Thou", 20, model, max_sequence_len))
print ("3. ",generate_text("King is", 20, model, max_sequence_len))
print ("4. ",generate_text("Death of", 20, model, max_sequence_len))
print ("5. ",generate_text("The Princess", 20, model, max_sequence_len))
print ("6. ",generate_text("Thanos", 20, model, max_sequence_len))
