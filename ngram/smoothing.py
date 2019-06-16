from typing import List
from ntlk.corpus import words

def get_ntlk_words() -> List[str]:
    """
    Get English words from NLTK corpus.
    """
    return words.words()
