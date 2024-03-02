import argparse
import logging

import numpy as np

from alignment.model_representations.model_representations import ModelRepresentations

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_words_file", type=str, default="/proj/inductive-bias.shadow/abakalov.data/words_fmri.npy")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--context_len", type=int, default=50)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)


    words = np.load(args.path_to_words_file)
    representations_calculator = ModelRepresentations(args.device, args.context_len)
    representations_calculator.compute_and_save_model_representations(words, args.output_dir)