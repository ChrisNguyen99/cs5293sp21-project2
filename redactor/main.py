import glob
import tempfile
import re
import io
import os
import pdb
import sys
import sklearn
import spacy
import nltk
import argparse
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag
from nltk import ne_chunk

from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import cross_val_score

#redact names based on person
def redact_names(txt):
    nlp = spacy.load('xx_ent_wiki_sm')
    doc = nlp(txt)
    redact = []
    with doc.retokenize() as retokenizer:
        for entity in doc.ents:
            retokenizer.merge(entity)
    for token in doc:
        if token.is_upper and len(token) > 1 or token.ent_type_ == 'PER':
            for num in range(len(token)):
                redact.append("X")
            redact.append(" ")
        else:
            redact.append(token.text)
            redact.append(" ")
    return "".join(redact)

def redact(input):
    
    #nlp = spacy.load('en_core_web_sm')
    
    files = glob.glob(input)

    for file in files:
        fi = open(file)
        text = fi.read()
        #doc = nlp(text)
        #for entity in doc.ents:
            #print(entity.text, entity.label_)

        redact_doc1 = redact_names(text)
        #print(redact_doc1)

        #print(file[16:])
        outfile = open("redacted/" + file[16:], "w")
        outfile.write(redact_doc1)
        outfile.close()

nlp = None

def get_entity(text):
    str = []
    """Prints the entity inside of the text."""
    for sent in sent_tokenize(text):
        for chunk in ne_chunk(pos_tag(word_tokenize(sent))):
            if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
                str.append(c[0] for c in chunk.leaves())
    return str

def doextraction(glob_text):
    arra = []
    """Get all the files from the given glob and pass them to the extractor."""
    for thefile in glob.glob(glob_text):
        with io.open(thefile, 'r', encoding='utf-8') as fyl:
            text = fyl.read()
            #arra = get_entity(text)
    return text

def make_features(sentence, ne="PERSON"):
    doc = nlp(sentence)
    D = []
    for e in doc.ents:
        if e.label_ == ne:
            d = {}
            # d["name"] = e.text # We want to predict this
            d["length"] = len(e.text)
            d["word_idx"] = e.start
            d["char_idx"] = e.start_char
            d["spaces"] = 1 if " " in e.text else 0
            # gender?
            # Number of occurences?
            D.append((d, e.text))
    return D


def main():
    # print(len(sample))
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="input txt files")
    
    args = parser.parse_args()
    #redact(args.input)
    doc = doextraction("../text/*.txt")
    print(doc)
    #for entities in doc:
        #features = []
        #for s in entities:
            #features.extend(make_features(s))
        #print(features)

    #v = DictVectorizer(sparse=False)
    #train_X = v.fit_transform([x for (x,y) in features[:-1]])
    #train_y = [y for (x,y) in features[:-1]]

    #test_X = v.fit_transform([x for (x,y) in features[-1:]])
    #test_y = [y for (x,y) in features[-1:]]

    #clf = DecisionTreeClassifier(criterion="entropy")
    #clf = KNeighborsClassifier(n_neighbors=3)
    #clf.fit(train_X, train_y)

    #print("Decison Tree: ", clf.predict(test_X), clf.predict_proba(test_X), test_y)

    #print("Cross Val Score: ", cross_val_score(clf,
                                               #v.fit_transform([x for (x,y) in features]),
                                               #[y for (x,y) in features],
                                               #cv=2))



if __name__ == "__main__":
    nlp = spacy.load("en_core_web_lg")
    main()
