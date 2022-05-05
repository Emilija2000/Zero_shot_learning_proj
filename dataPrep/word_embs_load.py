import numpy as np
import os
from scipy.io import loadmat

import data_utils

if __name__ == '__main__':

    config = data_utils.load_config()

    word_rep_file = os.path.join(config['DATASET']['word_ftrs_path_original'], config['DATASET']['word_ftrs_file_name_original'])
    word_rep_var_name = config['DATASET']['word_ftrs_var_name_original']
    word_vocab_file = os.path.join(config['DATASET']['word_ftrs_path_original'], config['DATASET']['word_vocab_file_name_original'])
    word_vocab_var_name = config['DATASET']['word_vocab_var_name_original']
    words = config['DATASET']['classes']

    # load pretrained word representations
    f = loadmat(word_rep_file)
    w = f[word_rep_var_name]

    # load vocabulary
    f = loadmat(word_vocab_file)
    vocab = f[word_vocab_var_name]

    # find key word reps
    ftrs = np.array([w[:,np.where(vocab==words[i])[1]].flatten() for i in range(len(words))])

    # save embs
    data_utils.save_data(config['DATASET']['word_ftrs_path'],ftrs)
    








