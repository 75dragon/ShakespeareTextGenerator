from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import keras.utils as ku
import pickle
import string, os, sys
tokenizer = None
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

max_sequence_len = 34
Pickled_Model = None
Pkl_Filename = "Pickle_RL_Model.pkl"
with open(Pkl_Filename, 'rb') as file:
    Pickled_Model = pickle.load(file)
def generate_text(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)

        output_word = ""
        for word,index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text.title()
print ("1. ",generate_text("Then", 20, Pickled_Model, max_sequence_len))
