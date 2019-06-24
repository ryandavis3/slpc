import os
import math

from collections import Counter
from typing import List, Set, Dict

import slpc.textnorm.preprocessing as tpr

## Directory and file names for example text files.

DATA_DIR = '/Users/ryandavis/Desktop/projects/slpc/data'
FN = {
        'amazon' : 'amazon_cells_labelled.txt',
        'imdb' : 'imdb_labelled.txt',
        'yelp' : 'yelp_labelled.txt'
}

def to_int(s: str) -> int:
    """
    If possible, convert string to int. Else, return None.
    """
    try:
        n = int(s)
    except ValueError:
        n = None
    return n

def read_dataset(name: str, directory: str=DATA_DIR) -> List[List]:
    """
    Read dataset from disk into memory
    """
    # Read raw text line by line
    fn = os.path.join(directory, FN[name])
    with open(fn) as f:
        text = f.readlines()
    # Split text from label, sanitize labels
    text_parse = []
    for line in text:
        line = line.strip().split('\t')
        if len(line) != 2:
            continue
        line[1] = to_int(line[1])
        if line[1] is None:
            continue
        if line[1] not in set([0,1]):
            continue
        text_parse += [line]
    return text_parse

def preprocess_text(text: str) -> str:
    """
    Preprocess text.
    """
    text = tpr.filter_alphabetic_chars(text)
    text = text.lower()
    text = tpr.collapse_whitespace(text)
    return text

def preprocess_texts(texts: List[str]) -> str:
    """
    Preprocess list of texts.
    """
    texts = [preprocess_text(text) for text in texts]
    texts = " ".join(texts)
    texts = tpr.collapse_whitespace(texts)
    return texts

def get_vocab_from_data(data: List[List]) -> Set:
    """
    Get vocabulary from data.
    """
    texts = [row[0] for row in data]
    texts = preprocess_texts(texts)
    words = texts.split(' ')
    V = set(words)
    return V

def get_word_counts(texts: List[str]):
    """
    Get word counts for group of texts.
    """
    texts = preprocess_texts(texts)
    words = texts.split(' ')
    return Counter(words)

class NaiveBayes:
    """
    Class implementing Naive Bayes for document classification.
    """
    def __init__(self):
        pass
    
    def train(self, data: List[List]):
        """
        Train classifier on text.
        """
        Ndoc = len(data)
        # Represent using dictionary by class
        bigdoc = {}
        for L in data:
            text = L[0]
            label = L[1]
            if label not in bigdoc:
                bigdoc[label] = []
            bigdoc[label] += [text]
        # Get log prior probababilities. Use MLE - estimate is the
        # relative frequency of the document.
        logprior = {}
        for label in bigdoc:
            logprior[label] = len(bigdoc[label]) / Ndoc
        # Get vocabulary
        V = get_vocab_from_data(data)
        # Word counts in each class
        bigdoc_count = {}
        for label in bigdoc:
            bigdoc_count[label] = get_word_counts(bigdoc[label])
        # Compute log likelihood for each class and word
        log_likelihood = {}
        labels = set(bigdoc)
        for label in labels:
            log_likelihood[label] = {}
            den = NaiveBayes.likelihood_denominator(V, bigdoc_count, label)
            for w in V:
                ct = NaiveBayes.get_smoothed_count(bigdoc_count[label], w) 
                log_likelihood[label][w] = math.log(ct / den)
        # Save data members
        self.logprior = logprior
        self.log_likelihood = log_likelihood
        self.V = V

    @staticmethod
    def get_smoothed_count(count: Dict, w: str) -> int:
        """
        Get smoothed count of word (add 1).
        """
        if w in count:
            return count[w] + 1
        else:
            return 1

    @staticmethod
    def likelihood_denominator(V: Set, bigdoc_count: Dict, label: int) -> float:
        """
        Compute denominator of likelihood.
        """
        den = 0.0
        for w in V:
            den += NaiveBayes.get_smoothed_count(bigdoc_count[label], w)
        return den
