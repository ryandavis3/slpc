#!/bin/python3

import re
from typing import List, Dict, Set

TRANSCRIPT_PATH = "data/transcript.txt"

def read_transcript_text(filename: str=TRANSCRIPT_PATH) -> str:
    """
    Read raw transcript from text file.

    Args:
        filename (str): File name from which to read. 

    Returns:
        list of str
    """
    with open(filename) as f:
        text = f.readlines()
    return text

def remove_leading_code(line: str) -> str:
    """
    Remove leading alphanumeric code from line.

    Args:
        line (str): Line of text from transcript.

    Returns:
        str

    Example:
        '33_1_0001 hello' -> 'hello'
    """
    space = line.find(" ")
    return line[space+1:]

def remove_extraenous_phrases(words: List[str]) -> List[str]:
    """
    Remove extraneous phrases from collection of words.

    Args:
        words (list of str)

    Returns:
        list of str
    """
    pattern = '[\[\<](.*?)[\]\>]'
    words = [word for word in words if re.search(pattern, word) is None]
    words = [word for word in words if word != '.']
    return words

def parse_words_from_line(line: str) -> List[str]:
    """
    Parse words from line of text.
    """
    line = remove_leading_code(line)
    words = line.split()
    words = remove_extraenous_phrases(words)
    words = ['<s>'] + words + ['</s>']
    return words

def ngram_search_backward(D: Dict, ngram: List[str], n: int, k: int) -> Dict:
    """
    Recursively build ngram nested dictionary backward (last word
    in n-gram at top level).
    """
    word = ngram[k]
    if k == 0:
        if word not in D:
            D[word] = 0
        D[word] += 1
    else:
        if word not in D:
            D[word] = {}
        D[word] = ngram_search_backward(D[word], ngram, n, k-1)
    return D

def ngram_search_forward(D: Dict, ngram: List[str], n: int, k: int) -> Dict:
    """
    Recursively build ngram nested dictionary forward (first word
    in n-gram at top level).
    """
    word = ngram[k]
    if k == n-1:
        if word not in D:
            D[word] = 0
        D[word] += 1
    else:
        if word not in D:
            D[word] = {}
        D[word] = ngram_search_forward(D[word], ngram, n, k+1)
    return D

def build_ngram_dict(text: str, n: int=2, reverse: bool=False) -> Dict:
    """
    Build nested dictionary representing ngrams in text. In the case
    of n=1, function returns a dcitionary with counts of each word.

    Args:
        text (list of str): Lines of text.
        n (int): Number of words per n-gram.
        reverse (bool): If True, use "reverse" order.

    Returns:
        dict: Nested keys represent subsequent words
            in an n-gram. Leaf values represent the count of the
            n-gram in the text.
    """
    D = {}
    for line in text:
        words = parse_words_from_line(line)
        ngrams = [words[i:i+n] for i in range(len(words)-n)]
        for ngram in ngrams:
            if reverse:
                D = ngram_search_backward(D, ngram, n, n-1)
            else:
                D = ngram_search_forward(D, ngram, n, 0)
    return D

def get_bigram_count_matrix_words(bigrams: Dict, words: List[str]) -> List[List[str]]:
    """
    Get counts of bigrams in matrix format for a given set of words.
    """
    L = [[''] + words]
    W = len(words)
    # First word
    for i, word1 in enumerate(words):
        Lw = [word1] + [None] * W
        # Second word
        for j, word2 in enumerate(words):
            if word1 not in bigrams:
                Lw[j+1] = 0
            elif word2 not in bigrams[word1]:
                Lw[j+1] = 0
            else:
                Lw[j+1] = bigrams[word1][word2]
        L += [Lw]
    return L

def get_subset_ngrams(ngrams: Dict, words: List[str], N: int, fill: bool=False) -> Dict:
    """
    Get subset of ngrams only including certain words. 
    """
    ngram_sub = {}
    # Unigram case -> return dictionary
    if N == 1:
        for word in words:
            if word in ngrams:
                ngram_sub[word] = ngrams[word]
            elif fill:
                ngram_sub[word] = 0
        return ngram_sub
    # N > 1 -> recursively get subset on smaller n-gram sets
    else:
        for word in words:
            if word in ngrams:
                ngram_sub[word] = get_subset_ngrams(ngrams[word], words, N-1, fill)
            elif fill:
                ngram_sub[word] = get_subset_ngrams({}, words, N-1, fill)
    return ngram_sub

def get_words(ngrams: Dict, n: int, inner: bool=False) -> Set:
    """
    Get all words in n-grams.`
    """
    if inner:
        words = set()
    else:
        words = set(ngrams)
    if n == 1:
        return set(ngrams)
    else:
        for word in ngrams:
            words = words.union(get_words(ngrams[word], n-1))
    return words
