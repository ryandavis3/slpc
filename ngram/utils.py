#!/bin/python3

import re

TRANSCRIPT_PATH = "data/transcript.txt"

def read_transcript_text(filename=TRANSCRIPT_PATH):
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

def remove_leading_code(line):
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

def remove_extraenous_phrases(words):
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

def build_ngram_dict(text, n=2):
    """
    Build nested dictionary representing ngrams in text.

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
                


    
