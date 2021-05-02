# cs5293sp21-project2
CS5293 Text Analytics Project 2
Unredact redacted files based on trained data. Target names.
python3 on linux, argparse, tmepfile, re, sys, glob, spacy, sklearn, nltk should be installed
Files in the textredacted folder are redacted intially. Files in the text folder are trained for use of DictVectorizer. KNeighborsClassifier is fitted for prediction on names.
The unredaction process takes the files redacted and find the best prediction for the redacted names which are outputed. Unredacted files are in the output folder.

Instructions:
get code using git clones: git@github.com:ChrisNguyen99/cs5293sp21-project1.git
run using: pipenv run python main.py --input '../textredacted/*.txt'
Here the input flag specifies where to find files for redacting files
Check unredacted files in output folder

Bugs/Assumptions:
spacy is installed and used for the redaction process
xx_ent_wiki_sm is installed under spacy and loaded in
X is the redacting character
tests are using single file
Not all "names" may be redacted as the pipeline may not have caught them
Unredacted names may be duplicated in the same file
Each text file has sentences in each of them with names
The original folder of data was very large and took too much time to parse. Number of files has been reduced to use
Grabbing features looks at length, word id, character id, and spaces

Testing:
For installation of pytest: pipenc install pytest
For running tests: pipenv run python -m pytest
Within test file functions are tested as follows:
test_redact_names() checks names were redacted to a nonempty file
test_redact() checks files were redacted to a nonempty file
test_unredact() checks redacted files were unredacted to a nonempty file

Sources:
official documentation:
https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html#sklearn.feature_extraction.DictVectorizer.fit_transform
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

https://stackoverflow.com/questions/4945548/remove-the-first-character-of-a-string: removing extraneous file path
https://stackoverflow.com/questions/46290313/how-to-break-up-document-by-sentences-with-with-spacy: using a sentencizer
https://ai.stanford.edu/~amaas/data/sentiment/: data files
