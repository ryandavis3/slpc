#!/bin/python3

import re
from typing import List, Dict

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

def build_ngram_dict(text: str, n: int=2) -> Dict:
    """
    Build nested dictionary representing ngrams in text. In the case
    of n=1, function returns a dcitionary with counts of each word.

    Args:
        text (list of str): Lines of text.
        n (int): Number of words per n-gram.

    Returns:
        dict: Nested keys represent subsequent words
            in an n-gram. Leaf values represent the count of the
            n-gram in the text.
    """
    ngram_dict = dict()
    for line in text:
        line = remove_leading_code(line)
        words = line.split()
        words = remove_extraenous_phrases(words)
        words = ['<s>'] + words + ['</s>']
        ngrams = [words[i:i+n] for i in range(len(words)-n)]
        for ngram in ngrams:
            dict_level = ngram_dict
            for i in range(n):
                word = ngram[i]
                if i < n-1:
                    if word not in dict_level:
                        dict_level[word] = dict()
                    dict_level = dict_level[word]
                else:
                    if word not in dict_level:
                        dict_level[word] = 0
                    dict_level[word] += 1
    return ngram_dict

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


