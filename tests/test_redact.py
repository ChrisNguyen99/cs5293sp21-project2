from __future__ import unicode_literals, print_function
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
import numpy as np
import fileinput
import random
import os
from redactor import main
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag
from nltk import ne_chunk

from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import cross_val_score

from spacy.lang.en import English

file = "1_7.txt"
file2 = "redacted/1.7.txt"
fi = open(file)
text = fi.read()

sample = ["""Rare is the red carpet where a single look summarizes the entire event. But gaze upon Lakeith Stanfield, with freshly auburn hair, in a Saint Laurent jumpsuit with a plunging v-neck and a pointy white collared shirt. It was super ’70s, with Stanfield’s belted waist and broad shoulders and dagger-point collar. But it was also titillatingly fluid. Kinda racy. Very sexy. He looked hot, weird, and charismatic. Up for Best Supporting Actor for his work in Judas and the Black Messiah, he looked like an artist. He also just looked awesome. It was like a treatise on the greater state of men’s style: this is what’s going on here, now.""",
        """If past genderfluid styles have been gauntlet-throwing statements of glamour, like Billy Porter’s velvet tuxedo gown in 2019, Stanfield’s was subtler, which is a nice reading of the room, but it was also a more provocative use of fashion. “I wanted to express who he is as a person: someone who is equally thoughtful as he is playful,” his stylist, Julie Ragolia, explained in a text message. Saint Laurent designer Anthony Vaccarello’s Spring 2021 women’s collection “stayed with me,” she said, and they decided to adapt a piece from it, a lean jumpsuit that recalls the eponymous designer’s fondness for safari jackets, for Stanfield: “In thinking of a way to balance the formality of such a show, this special nomination for LaKeith, and the seriousness of the times we are all living in, coming to such a look just felt thoughtful, while still being celebratory.” Ragolia also noted that the look was made with sustainable materials.""",
        """If there’s one look from the Oscars red carpet last night that got people talking, it was actor LaKeith Stanfield’s gender-bending, perfectly fitting, super-sexy Saint Laurent jumpsuit. He may not have won the award for Best Supporting Actor for his role in Judas and the Black Messiah, but he took home the prize for Best Dressed in my book.""",
        """FBI informant William O’Neal (LaKeith Stanfield) infiltrates the Illinois Black Panther Party and is tasked with keeping tabs on their charismatic leader, Chairman Fred Hampton (Daniel Kaluuya). A career thief, O’Neal revels in the danger of manipulating both his comrades and his handler, Special Agent Roy Mitchell. Hampton’s political prowess grows just as he’s falling in love with fellow revolutionary Deborah Johnson. Meanwhile, a battle wages for O’Neal’s soul. Will he align with the forces of good? Or subdue Hampton and The Panthers by any means, as FBI Director J. Edgar Hoover commands?"""]

v = DictVectorizer(sparse=False)
clf = DecisionTreeClassifier(criterion="entropy")

def test_redact_names():
    redact_doc1 = main.redact_names(text)
    assert len(redact_doc1) > 1

def test_redact():
    redact_doc2 = main.redact(file2)
    outfile = open("redacted/" + file, "w")
    assert os.stat("redacted/1_7.txt").st_size == 0

def test_make_features():
    features = []
    for thefile in glob.glob("*.txt"):
        with io.open(thefile, 'r', encoding='utf-8') as fyl:
            text = fyl.read()
            nlp = English()
            nlp.add_pipe('sentencizer')
            doc = nlp(text)
            sentences = [str(sent).strip() for sent in doc.sents]
    for s in sentences:
        features.extend(main.make_features(s))
    assert len(features) > 1

def test_train():
    features = []
    features = main.train(clf, v, features)
    assert len(features) > 1

def test_unredact():
    main.unredact(clf, v)
    outfile = open("output/" + file, "w")
    assert os.stat("output/1_7.txt").st_size == 0
