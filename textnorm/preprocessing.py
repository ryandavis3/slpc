import re

def filter_alphabetic_chars(s: str) -> str:
    """
    Remove all non-alphanumeric characters and non-spaces 
    from string.
    """
    return re.sub(r'[^a-zA-Z\s]', '', s)

def collapse_whitespace(s: str) -> str:
    """
    Collapse multiple consecutive  whitespaces to a single 
    whitespace.
    """
    return re.sub(' +', ' ', s)
