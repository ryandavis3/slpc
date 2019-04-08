import re 
import collections

## Implement byte-pair encoding (BPE) learning algorithm

def get_stats(vocab):
    """
    Get frequency of all pairs of characters in vocabulary.

    Args:
        vocab (dict): Keys are words. Values are frequencies of 
            words. Words must have spaces between characters
            and end with </w>.

    Returns:
        pairs (defaultdict): Each key is a tuple of two characters
            e.g. ('l', 'o'). Each value is an integer of the frequency
            that the pair of characters appears.
    """
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    """
    Merge vocabulary with new combined pair of characters.

    Args:
        pair (tuple of two str): Pair to be treated as a single
            character.
        v_in (dict): Keys are words. Values are frequencies of
            words. Words must have spaces between characters
            and end with </w>.

    Returns:
        dict: Vocabulary with 'pair' merged in.
    """
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out
