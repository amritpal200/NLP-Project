import spacy

import pycrfsuite as crfs

import json
import os

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from itertools import chain


# Function to read sentences from a file and store them in a list
def read_file(file_path):
    sentences = []
    with open(file_path, 'r') as file:
        for line in file:
            sentences.append(line.strip())
    return sentences

# Function to process text data
def process_text(data):
    final_sentences=[]
    final_sentences_chars = []

    # Iterate through each data entry
    for i in range(len(data)):
        text = data[i]['data']['text']
        chars = list(text)

        # Iterate through the predictions for each entry
        for value in data[i]['predictions'][0]['result']:
            index = (value['value']['start'], value['value']['end'])
            label = value['value']['labels'][0]

            # Update characters to reflect labels
            for j in range(index[0], index[1]-1):
                if(chars[j] not in " ."): chars[j] = label
        original_text = ''.join(chars)

        # Split text into sentences
        sentences = [sentence.strip() + '.' for sentence in text.split(". ")]
        final_sentences.extend(sentences)

        # Split labeled text into sentences
        sentences_chars= [sentence.strip() + '.' for sentence in original_text.split(". ")]
        final_sentences_chars.extend(sentences_chars)

    return final_sentences, final_sentences_chars

# Function to convert sentences into token features
def from_sentences_to_tokenfeatures(sentences, nlp):
    feature_sentences=[]

    # Process each sentence
    for sentence in sentences:
        feature_words=[]

        # Process the sentence with SpaCy
        doc = nlp(sentence)

        # Extract base forms, part-of-speech tags, chunk tags, and named entity tags
        base_forms = [token.lemma_ for token in doc]
        pos_tags = [token.pos_ for token in doc]
        chunk_tags = [token.dep_ for token in doc]

        # Output the results in a tab-separated format
        for token, lemma, pos, chunk in zip(doc, base_forms, pos_tags, chunk_tags):
            feature_words.append([token.text, lemma, pos, chunk])
        feature_sentences.append(feature_words)
    return feature_sentences

# Function to convert sentences into BIO tagging
def from_sentences_to_BIO_tagging(sentences, sentences_char, nlp): 
    tagged_sentences=[]

    # Iterate through each sentence and its corresponding labeled characters
    for sentence, sentence_char in zip(sentences, sentences_char):
        doc1 = nlp(sentence)
        doc2 = nlp(sentence_char)
        sentence_list=[token.text for token in doc1]
        sentence_char_list=[token.text for token in doc2]
        tagged_sentence=['O'] * len(sentence_list)

        # Iterate through each word and its labeled character
        for i, (word, word_char) in enumerate(zip(sentence_list, sentence_char_list)):
            
            # Apply BIO tagging based on labels
            if("NEG" in word_char):
                if(i!=0 and (tagged_sentence[i-1]=='B-NEG' or tagged_sentence[i-1]=='I-NEG')): tagged_sentence[i]='I-NEG'
                else: tagged_sentence[i]='B-NEG'
            elif("UNC" in word_char):
                if(i!=0 and (tagged_sentence[i-1]=='B-UNC' or tagged_sentence[i-1]=='I-UNC')): tagged_sentence[i]='I-UNC'
                else: tagged_sentence[i]='B-UNC'
            elif("NSCO" in word_char):
                if(i!=0 and (tagged_sentence[i-1]=='B-NSCO' or tagged_sentence[i-1]=='I-NSCO')): tagged_sentence[i]='I-NSCO'
                else: tagged_sentence[i]='B-NSCO'
            elif("USCO" in word_char):
                if(i!=0 and (tagged_sentence[i-1]=='B-USCO' or tagged_sentence[i-1]=='I-USCO')): tagged_sentence[i]='I-USCO'
                else: tagged_sentence[i]='B-USCO'
        tagged_sentences.append(tagged_sentence)
    return tagged_sentences

# Define feature extraction function
def token2features(tokens, i):
    token = tokens[i]
    features = {
        'token': token[0],        # The token itself
        'lemma': token[1],        # Lemma
        'pos_tag': token[2],      # POS tagging
        'chunk': token[3],        # Chunk
        'bias': 1.0
    }
    if i > 0:
        prev_token = tokens[i-1]
        features.update({
            'prev_token': prev_token[0],
            'prev_pos_tag': prev_token[2],
            'prev_chunk': prev_token[3]
        })
    else:
        features['BOS'] = True  # Beginning of sequence
    if i < len(tokens)-1:
        next_token = tokens[i+1]
        features.update({
            'next_token': next_token[0],
            'next_pos_tag': next_token[2],
            'next_chunk': next_token[3]
        })
    else:
        features['EOS'] = True  # End of sequence
    return features

# Function to prepare data
def prepare_data(X, Y):
    X_data = []
    y_data = []
    for sentence_tokens, sentence_labels in zip(X, Y):
        X_sentence = [token2features(sentence_tokens, i) for i in range(len(sentence_tokens))]
        y_sentence = sentence_labels
        X_data.append(X_sentence)
        y_data.append(y_sentence)
    return X_data, y_data

# Train the CRF model
def train_model(X_train, y_train):
    trainer = crfs.Trainer(verbose=False)
    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)
    trainer.set_params({'c1': 1.0, 'c2': 1e-3, 'max_iterations': 50, 'feature.possible_transitions': True})
    trainer.train('../data/bio_crf.crfsuite')

def bio_classification_report(y_true, y_pred):
    """

    Classification report.
    You can use this as evaluation for both in the baseline model and new model.

    """
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset,
    )
