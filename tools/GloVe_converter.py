#https://nlp.stanford.edu/projects/glove/

from gensim.scripts.glove2word2vec import glove2word2vec
from src.data_dir_config import root

glove_input_file = root + 'res/glove/glove.840B.300d.txt'
word2vec_output_file = root + 'res/glove/glove.840B.300d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)