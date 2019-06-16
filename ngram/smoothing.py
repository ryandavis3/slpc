from typing import List, Dict, Set

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

def laplace_smooth_unigram(count: Dict, vocab: List[str]) -> Dict:
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
