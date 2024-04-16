import numpy as np
import tensorflow as tf
import pickle
import spacy
import en_core_web_lg
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the trained model
model = load_model('next_word_model.h5')  # Update the path to your model

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Ensure you have the correct max_sequence_len used during training
max_sequence_len = 20  # Update this with the actual value used during training

def predict_next_words(text, n=3):
    sequence = tokenizer.texts_to_sequences([text])[0]
    sequence = pad_sequences([sequence], maxlen=max_sequence_len - 1, padding='pre')
    predictions = model.predict(sequence, verbose=0)[0]

    top_indices = np.argsort(predictions)[-n:][::-1]
    top_words = []
    for i in top_indices:
        for word, index in tokenizer.word_index.items():
            if index == i:
                top_words.append(word)
                break

    return top_words

# Load spaCy's English language model
nlp = en_core_web_lg.load()

def find_synonyms(word, vocab, threshold=0.9):
    word_doc = nlp(word)
    # Check if the input word is an article/determiner; if so, return empty list
    if word_doc[0].tag_ == 'DT' or word_doc[0].tag_ == 'PRP':
        print(word)
        return []
    synonyms = []
    for v_word in vocab:
        v_word_doc = nlp(v_word)
        # Skip vocabulary words that are articles/determiners
        if (v_word_doc[0].tag_ != 'DT' or v_word_doc[0].tag_ != 'PRP') and word_doc.similarity(v_word_doc) > threshold:
            synonyms.append(v_word)
    return synonyms

def predict_next_words_with_synonyms(text, vocabulary, n=3, threshold=0.55):
    top_words = predict_next_words(text, n)
    synonyms_in_vocab = {}

    for word in top_words:
        synonyms = find_synonyms(word, vocabulary, threshold)
        synonyms_in_vocab[word] = synonyms if synonyms else []

    return dict(sorted(synonyms_in_vocab.items()))

text = "You've got a broken heart, and maybe it could fix"
vocabulary = ["inform", "speak", "complication", "cherish", "admire", "value", "pixie", "gorilla", "elf", "find", "research", "detection", "goblin", "gnome", "language", "society", "globe", "planet", "fondness", "love", "emotion", "sentiment", "top", "most", "elite"]

# Example usage
print(predict_next_words(text, 10))
print(predict_next_words_with_synonyms(text, vocabulary, 10, 0.52821))

#print(find_synonyms('the', vocabulary, 0.5))
# print(predict_next_words_with_synonyms(text, vocabulary, 10, 0.5866))
