import argparse
import logging

import slpc.textnorm.text_normalization as tn

def _main():
    """
    Run the byte-pair encoding (BPE) algorithm.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_merges", type=int, default=8)
    parser.add_argument("--log", "-log", type=str, default='INFO',
                        choices=['INFO', 'DEBUG', 'WARNING', 'ERROR'])
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log))
    # Define initial vocabulary
    vocab = {
        ' l o w </w>' : 5 , 
        ' l o w e s t </w>' : 2 ,
        ' n e w e r </w>' : 6 , 
        ' w i d e r </w>' : 3 , 
        ' n e w </w>' : 2 
    }
    # Iteratively merge new characters
    for i in range(args.num_merges):
        logging.info("Performing merge %s", i)
        pairs = tn.get_stats(vocab)
        best = max(pairs, key=pairs.get)
        vocab = tn.merge_vocab(best, vocab)
        logging.info("Merge pair %s", best)
    logging.info("Merged vocabulary: %s", vocab)

if __name__ == "__main__":
    _main()
