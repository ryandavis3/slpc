import logging
from typing import List, Dict, Set

import slpc.ngram.utils as nu

def get_ntlk_words() -> List[str]:
    """
    Get English words from NLTK corpus.
    """
    from nltk.corpus import words
    return words.words()

def remove_words_outside_vocab(ngrams: Dict, vocab) -> Dict:
    """
    Remove words in ngram dictionary outside vocabulary.
    """
    if isinstance(vocab, list):
        vocab = set(vocab)
    remove = []
    for word in ngrams:
        if word not in vocab:
            remove += [word]
    for word in remove:
        del ngrams[word]
    return ngrams

def expand_vocab(vocab, corpus) -> Set:
    """
    Expand vocabulary to include new words from corpus.

    Args:
        vocab (List or Set)
        corpus (Dict, List, or Set)
    """
    if isinstance(vocab, list):
        vocab = set(vocab)
    if isinstance(corpus, dict):
        corpus = set(corpus.keys())
    elif isinstance(corpus, list):
        corpus = set(corpus)
    vocab = vocab.union(corpus)
    return vocab

def laplace_smooth_unigram_prob(count: Dict, vocab: List[str]) -> Dict:
    """
    Apply Laplace smoothing to generate probabilities from word
    counts.
    """
    # Expand vocabulary to include words in 'count'
    vocab = expand_vocab(vocab, count)
    # V is the number of words in the vocabulary
    V = len(vocab)
    # N is the number of word tokens in the corpus
    N = sum(count.values())
    prob = dict()
    # Add one to each count (including previously zero counts)
    # and compute probability.
    for word in vocab:
        if word in count:
            cw = count[word]
        else:
            cw = 0
        prob[word] = (cw + 1) / (N + V)
    return prob

def add_one(D: Dict, word: str) -> Dict:
    """
    Increment value in dictionary if it exists. If it does not exist,
    set value to 1.
    """
    if word in D:
        D[word] += 1
    else:
        D[word] = 1
    return D

def laplace_smooth_count(count: Dict, vocab: Set, n: int) -> Dict:
    """
    Perform Laplace smoothing on n-gram counts. Add one to count
    of each n-gram and set n-grams with zero count to one.
    """
    count = count.copy()
    # n = 1 -> increment count and return
    if n == 1:
        for word in vocab:
            count = add_one(count, word)
        return count
    # n > 1 -> recursively smooth on smaller n-grams
    else:
        for word in vocab:
            if word not in count:
                count[word] = {}
            count[word] = laplace_smooth_count(count[word], vocab, n-1)
    return count

def laplace_smooth_prob_search(count: Dict, vocab: Set, n: int, fills: Dict) -> Dict:
    """
    Recursively get Laplace smoothed probability for n-grams.
    """
    P = {}
    if n == 1:
        for word in vocab:
            count = add_one(count, word)
        C = sum(count.values())
        for word in vocab:
            P[word] = count[word] / C
        return P
    else:
        for word in vocab:
            if word in count:
                P[word] = laplace_smooth_prob_search(count[word], vocab, n-1, fills)
            else:
                P[word] = fills[n-1]
    return P

def laplace_smooth_prob(count: Dict, n: int, vocab: Set=None) -> Dict:
    """
    Get Laplace smoothed probability for each ngram.
    
    If vocabulary not given, use unique words in ngrams as vocabulary.
    """
    if not vocab:
        vocab = nu.get_words(count, n)
    fills = {}
    for k in range(1, n+1):
        fills[k] = laplace_fill_probability(vocab, k)
    P = laplace_smooth_prob_search(count, vocab, n, fills)
    return P

## TODO: Implement EM algorithm to fit parameters of interpolation.

def laplace_fill_probability(vocab: Set, n: int):
    """
    Fill probabilities where none of the vocabulary is present.
    
    TODO: Can optimize with different data structure that recognizes
    if word / n-gram is missing rather than explicitly storing the
    entire vocabulary.
    """
    P = {}
    if n == 1:
        p = 1 / len(vocab)
        for word in vocab:
            P[word] = p
    else:
        filler = laplace_fill_probability(vocab, n-1)
        for word in vocab:
            P[word] = filler
    return P

class Interpolation:
    """
    Class for simple linear interpolation using different order n-grams.
    """
    def __init__(self, text: str, n: int):
        """
        Constructor. Store n-gram dictionary as a data member.
        """
        self.n = n

        # Store counts for each k-gram for k = 1,2,...n.
        self.ngrams = {}
        for k in range(1, n+1):
            logging.info('Building n-gram dictionaries.')
            self.ngrams[k] = nu.build_ngram_dict(text, k)
        
        # Store probabilities for each k-gram for k = 1,2,...n.
        self.ngrams_prob = {}
        for k in range(1, n+1):
            logging.info('Building n-gram probabilities for n=%s.', k)
            self.ngrams_prob[k] = laplace_smooth_prob(self.ngrams[k], k)

        # Placeholder for parameters
        self.params = [None] * n

    def predict(self, ngram: List[str]) -> float:
        """
        Predict probability of an n-gram.
        """
        p = 0
        for k in range(self.n):
            P = self.ngrams_prob[k+1]
            j = 0
            while j < k:
                P = P[ngram[j]]
                j += 1
            p += self.params[k] * P[ngram[j]]
        return p

## TODO: Function to get continuation probability following JM page 54.

def get_continuation_probability(word: str, ngrams: Dict, n: int, W: int=None) -> float:
    """
    Get continuation probability for word. Base the continuation
    probability on the number of different contexts the word has
    appeared in (i.e. the number of bigrams it completes.
    
    Args:
        word (str)
        ngrams (Dict): Must be in reversed order!!!
    """
    if not W:
        words = nu.get_words(ngrams, n, inner=True)
        W = len(W)
    return len(ngrams[word]) / W


