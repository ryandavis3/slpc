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

def laplace_smooth_prob_search(count: Dict, vocab: Set, n: int) -> Dict:
    """
    Recursively get Laplace smoothed probability for n-grams.
    """
    P = {}
    if n == 1:
        C = sum(count.values())
        for word in vocab:
            P[word] = count[word] / C
        return P
    else:
        for word in vocab:
            P[word] = laplace_smooth_prob_search(count[word], vocab, n-1)
    return P

def laplace_smooth_prob(count: Dict, n: int, vocab: Set=None) -> Dict:
    """
    Get Laplace smoothed probability for each ngram.
    
    If vocabulary not given, use unique words in ngrams as vocabulary.
    """
    if not vocab:
        vocab = nu.get_words(count, n)
    count_smooth = laplace_smooth_count(count, vocab, n)
    P = laplace_smooth_prob_search(count_smooth, vocab, n)
    return P

## TODO: Implement class for linear interpolation. Implement
## EM algorithm to fit parameters of interpolation.

## TODO: Function to get continuation probability following JM page 54.


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
            self.ngrams[k] = nu.build_ngram_dict(text, k)
        # Store probabilities for each k-gram for k = 1,2,...n.
        self.ngrams_prob = {}
        for k in range(1, n+1):
            print(k)
            self.ngrams_prob[k] = laplace_smooth_prob(self.ngrams[k], k)


