# https://github.com/albertauyeung/python-crf-named-entity-recognition/blob/master/run.py
# http://www.albertauyeung.com/post/python-sequence-labelling-with-crf/
import codecs
import numpy as np
import nltk
import pycrfsuite
import sys
from bs4 import BeautifulSoup as bs
from bs4.element import Tag
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets

data = []
bagOfWords = []
for i in range(1, 16):
    doc = []
    f = open(str(i) + '.tsv', encoding='utf-8')
    prevLabel = ""
    for line in f:
        line = line.strip().split('\t')
        if len(line) > 2:
            if '#' not in line[0]:
                if line[3].startswith("Citation\_Introduction"):
                    label = "Citation_Introduction"
                    # label = "O"
                elif line[3].startswith("Citation"):
                    label = "Citation"
                else: # line[3].startswith("_"):
                    label = "O"
                if label == "Citation":
                    if prevLabel != label:
                        bagOfWords.append(line[2])
                    else:
                        lastEntry = bagOfWords[-1]
                        if lastEntry[-1] in ['(', ')', '.', ':'] or line[2] in [".", ":", '\'']:
                            bagOfWords[-1] = lastEntry + line[2]
                        else:
                            bagOfWords[-1] = lastEntry + " " + line[2]

                prevLabel = label
                doc.append((line[2], label))
    f.close()
    data.append(doc)

print(data)


def word2features(doc, i):
    word = doc[i][0]
    # TODO: Add features
    # Common features for all words
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
    ]

    # Features for words that are not
    # at the beginning of a document
    if i > 0:
        word1 = doc[i - 1][0]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:word.isdigit=%s' % word1.isdigit(),
        ])
    else:
        # Indicate that it is the 'beginning of a document'
        features.append('BOS')

    # Features for words that are not
    # at the end of a document
    if i < len(doc) - 1:
        word1 = doc[i + 1][0]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:word.isdigit=%s' % word1.isdigit(),
        ])
    else:
        # Indicate that it is the 'end of a document'
        features.append('EOS')

    return features


# A function for extracting features in documents
def extract_features(doc):
    return [word2features(doc, i) for i in range(len(doc))]


# A function fo generating the list of labels for each document
def get_labels(doc):
    return [label for (token, label) in doc]


X = [extract_features(doc) for doc in data]
y = [get_labels(doc) for doc in data]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

trainer = pycrfsuite.Trainer(verbose=True)

# Submit training data to the trainer
for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)

# Set the parameters of the model
trainer.set_params({
    # coefficient for L1 penalty
    'c1': 0.1,

    # coefficient for L2 penalty
    'c2': 0.01,

    # maximum number of iterations
    'max_iterations': 200,

    # whether to include transitions that
    # are possible, but not observed
    'feature.possible_transitions': True
})

# Provide a file name as a parameter to the train function, such that
# the model will be saved to the file when training is finished
trainer.train('crf.model')

# Generate predictions
tagger = pycrfsuite.Tagger()
tagger.open('crf.model')
y_pred = [tagger.tag(xseq) for xseq in X_test]

resp = datasets.load_files('../teshuva_classification/testclassifier/', encoding='utf-8')
# allOfBeisYosef = resp.data
allOfBeisYosef = []
allOfBeisYosefFeatures = []
lenBY = len(resp.data)
for i in range(lenBY):
    words = []
    for word in resp.data[i].split(" "):
        words.append((word, "None"))
    allOfBeisYosef += [words]
    allOfBeisYosefFeatures += [extract_features(words)]

# allOfBeisYosefFeatures = [extract_features(doc) for doc in allOfBeisYosef]
allOfBeisYosefTagged = [tagger.tag(siman) for siman in allOfBeisYosefFeatures]
BYWordsAndTags = []
for i in range(lenBY):
    BYWordsAndTags += [zip(allOfBeisYosef[i], allOfBeisYosefTagged[i])]
# BYWordsAndTags = zip(allOfBeisYosef, allOfBeisYosefTagged)

prevLabel = ""
newBagOfWords = []
for siman in BYWordsAndTags:
    for word in siman:
        if word[1] == "Citation":
            if prevLabel != "Citation":
                newBagOfWords.append(word[0][0])
            else:
                lastCitation = newBagOfWords[-1]
                if lastCitation[-1] in ['(', ')', '.', ':'] or word[0][0] in [".", ":", '\'']:
                    newBagOfWords[-1] = lastCitation + word[0][0]
                else:
                    newBagOfWords[-1] = lastCitation + " " + word[0][0]
        prevLabel = word[1]

# Let's take a look at a random sample in the testing set
i = 0
for x, y in zip(y_pred[i], [x[1].split("=")[1] for x in X_test[i]]):
    print("%s (%s)" % (y, x))

# Create a mapping of labels to indices
# TODO: make sure all labels are here
labels = {"O": 1, "Citation": 0, 'Citation_Introduction': 2}
# labels = {"O": 1, "Citation": 0}


# Convert the sequences of tags into a 1-dimensional array
predictions = np.array([labels[tag] for row in y_pred for tag in row])
truths = np.array([labels[tag] for row in y_test for tag in row])

# Print out the classification report
print(classification_report(
    truths, predictions,
    target_names=["O", "Citation", "Citation_Introduction"]))
    # target_names=["O", "Citation"]))

print('predictions', predictions)
print('truths', truths)
print(y_pred)
print(X_test[0])


bagofWordsFile = open("citationBagOfWords", "w", encoding="utf-8")
for word in newBagOfWords:
    bagofWordsFile.write(word + "\n")

bagofWordsFile.close()


