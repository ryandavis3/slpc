#!/bin/python3

import re

TRANSCRIPT_PATH = "slpc/data/transcript.txt"

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
    return [word for word in words if re.search(pattern, word) is None]
