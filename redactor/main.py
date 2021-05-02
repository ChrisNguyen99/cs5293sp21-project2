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

from spacy.lang.en import English

sample = ["""Rare is the red carpet where a single look summarizes the entire event. But gaze upon Lakeith Stanfield, with freshly auburn hair, in a Saint Laurent jumpsuit with a plunging v-neck and a pointy white collared shirt. It was super ’70s, with Stanfield’s belted waist and broad shoulders and dagger-point collar. But it was also titillatingly fluid. Kinda racy. Very sexy. He looked hot, weird, and charismatic. Up for Best Supporting Actor for his work in Judas and the Black Messiah, he looked like an artist. He also just looked awesome. It was like a treatise on the greater state of men’s style: this is what’s going on here, now.""",
        """If past genderfluid styles have been gauntlet-throwing statements of glamour, like Billy Porter’s velvet tuxedo gown in 2019, Stanfield’s was subtler, which is a nice reading of the room, but it was also a more provocative use of fashion. “I wanted to express who he is as a person: someone who is equally thoughtful as he is playful,” his stylist, Julie Ragolia, explained in a text message. Saint Laurent designer Anthony Vaccarello’s Spring 2021 women’s collection “stayed with me,” she said, and they decided to adapt a piece from it, a lean jumpsuit that recalls the eponymous designer’s fondness for safari jackets, for Stanfield: “In thinking of a way to balance the formality of such a show, this special nomination for LaKeith, and the seriousness of the times we are all living in, coming to such a look just felt thoughtful, while still being celebratory.” Ragolia also noted that the look was made with sustainable materials.""",
        """If there’s one look from the Oscars red carpet last night that got people talking, it was actor LaKeith Stanfield’s gender-bending, perfectly fitting, super-sexy Saint Laurent jumpsuit. He may not have won the award for Best Supporting Actor for his role in Judas and the Black Messiah, but he took home the prize for Best Dressed in my book.""",
        """FBI informant William O’Neal (LaKeith Stanfield) infiltrates the Illinois Black Panther Party and is tasked with keeping tabs on their charismatic leader, Chairman Fred Hampton (Daniel Kaluuya). A career thief, O’Neal revels in the danger of manipulating both his comrades and his handler, Special Agent Roy Mitchell. Hampton’s political prowess grows just as he’s falling in love with fellow revolutionary Deborah Johnson. Meanwhile, a battle wages for O’Neal’s soul. Will he align with the forces of good? Or subdue Hampton and The Panthers by any means, as FBI Director J. Edgar Hoover commands?"""]

samp = ["""Bromwell High is a cartoon comedy . It ran at the same time as some other programs about school life , such as " Teachers " . My 35 years in the teaching profession lead me to believe that Bromwell High ' s satire is much closer to reality than is " Teachers " . The scramble to survive financially , the insightful students who can see right through their pathetic teachers ' pomp , the pettiness of the whole situation , all remind me of the schools I knew and their students . When I saw the episode in which a student repeatedly tried to burn down the school , I immediately recalled ......... at .......... High . A classic line : XXXXXXXXX : I'm here to sack one of your teachers . XXXXXXX : Welcome to Bromwell High . I expect that many adults of my age think that Bromwell High is far fetched . What a pity that it isn't ! """,
"""' XXXXXXXXX ' is a wonderfully made Australian film honouring a true Australian hero . We are taken into the world of XXX , his best friend , XXXXXXXXX , and the other members of the XXXXXXXXXX , as the film explains and perhaps justifies Ned's actions.<br /><br />There is an exceptional cast present , who all give stellar performances , which brings the film to life . ( Great job , Heath! ) XXXXXXXXXXXXX was fantastic as XXX , playing the role of quiet , loyal ladiesman very well . I was swept up in the moment . For a moment I almost believed the Gang would win the battle at Glenrowan , alas , it was not to be.<br /><br />Some aspects of the film are fictional , and as an avid XXXXXXXXX fan ( and supporter ) , was slightly disappointed by this . Perhaps also the film could've gone longer , to cover more of the Kelly Gang's / Ned's life - I felt not enough was covered.<br /><br />Regardless of a few flaws , this is a moving film , which stirs all sorts of emotions . ( And hey , I'd assume this film would be better to watch rather than XXXXXXXXXXX trying to portray XXX ... ) """,
"""' They All Laughed ' is a superb XXXXXXXXXXXXXXXXX that is finally getting the recognition it deserves , and why ? their are many reasons the fact that it 's set in new york which truly sets the tone , the fantastic soundtrack , the appealing star turns from XXXXXXXXXXX , and the late XXXXXXXXXXX who is superb . and of course no classic is complete without XXXXXXXXXXXXXX . the film is a light and breezy romantic comedy that is very much in the vein of screwball comedy from the thirties , film is essentially about the Odyssey detective agency which is run by XXXXXXX who with his fellow detectives pot smoking and roller skating eccentric XXXXXX Novak(the films co - producer ) and XXXXXXXXXXX , basically the Gazzara falls for a rich tycoon magnate 's wife(Hepburn ) and XXXXXX falls for beautiful XXXXXXXXXXXXXXXX who sadly murdered infamously after production , ' They All Laughed is essential viewing for Bogdanovich fans . """,
"""XXXXXXXXXX ( yes , she must have ) know we would still be here for her some nine years later?<br /><br />See it if you haven't , again if you have ; see her live while you can . """]

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
    features = []
    for s in sample:
        features.extend(make_features(s))

    # print(features)

    v = DictVectorizer(sparse=False)
    train_X = v.fit_transform([x for (x,y) in features[:-1]])
    train_y = [y for (x,y) in features[:-1]]

    #test_X = v.fit_transform([x for (x,y) in features[-1:]])
    #test_y = [y for (x,y) in features[-1:]]

    clf = DecisionTreeClassifier(criterion="entropy")
    #clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(train_X, train_y)

    #print("Decison Tree: ", clf.predict(test_X), clf.predict_proba(test_X), test_y)

    # print(len(sample))
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="input txt files")
    
    args = parser.parse_args()
    #redact(args.input)

    docs = []
    #v = DictVectorizer(sparse=False)
    for thefile in glob.glob("../text/*.txt"):
        with io.open(thefile, 'r', encoding='utf-8') as fyl:
            text = fyl.read()
            nlp = English()
            nlp.add_pipe('sentencizer')
            doc = nlp(text)
            sentences = [str(sent).strip() for sent in doc.sents]
            #features = []
            #for entities in text:
            for s in sentences:
                features.extend(make_features(s))
            #print(features)

            #v = DictVectorizer(sparse=False)
            if [x for (x,y) in features[:-1]]:
                train_X = v.fit_transform([x for (x,y) in features[:-1]])
            #train_X = np.reshape(train_X, (-1, 1))
                train_y = [y for (x,y) in features[:-1]]
                clf.fit(train_X, train_y)

    for thefile in glob.glob("redacted/*.txt"):
        with io.open(thefile, 'r', encoding='utf-8') as fyl:
            text = fyl.read()
            nlp = English()
            nlp.add_pipe('sentencizer')
            doc = nlp(text)
            sentences = [str(sent).strip() for sent in doc.sents]
            #print(doc)
            feature = []
            ##for entities in text:
            for s in sentences:
                feature.extend(make_features(s))
            #print(feature)

            #temp = x for (x,y) in features[-1:]
            #if len(temp > 0):
            if feature:
                test_X = v.fit_transform([x for (x,y) in feature[-1:]])
                test_y = [y for (x,y) in feature[-1:]]

            #clf = DecisionTreeClassifier(criterion="entropy")
            #clf = KNeighborsClassifier(n_neighbors=3)
            #clf.fit(test_X, test_y)

                print("Decison Tree: ", clf.predict(test_X), test_y)

    #print("Cross Val Score: ", cross_val_score(clf,
                                               #v.fit_transform([x for (x,y) in features]),
                                               #[y for (x,y) in features],
                                               #cv=2))



if __name__ == "__main__":
    nlp = spacy.load("en_core_web_lg")
    main()
