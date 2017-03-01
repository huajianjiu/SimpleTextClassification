import numpy as np
import tensorflow as tf
import sys

WORD_VECTOR = {}
flags = tf.flags
logging = tf.logging
flags.DEFINE_string("vector_file", None, "vector_file")
flags.DEFINE_string("train_data", None, "train_data")
flags.DEFINE_string("test_data", None, "test_data")
FLAGS = flags.FLAGS


class Config(object):
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    vec_size = 200
    max_epoch = 6
    max_max_epoch = 59
    keep_prob = 0.5
    lr_decay = 0.8
    class_size = 1  # 1bit. 1 for positive, 0 for negative.


def read_vector_file(filename):
    with open(filename, "r") as f:
        word_vector_lines = f.readlines()
        for word_vector_line in word_vector_lines:
            if len(word_vector_line.split(" ")) < 3:
                continue
            word_vector_line = word_vector_line.split(" ")
            WORD_VECTOR[word_vector_line[0]] = np.array(word_vector_line[1:-1], dtype="float32")


def get_vector(w):
    try:
        return word_vector[w]
    except KeyError:
        return np.zeros(200)
    except:
        print "Unexpected error:", sys.exc_info()[0]
        return np.zeros(200)


class SoftmaxClassifier(object):
    def __init__(self, is_training=True, config=Config()):
        self._config = config
        self.vec_size = vec_size = config.vec_size
        self.class_size = class_size = config.class_size
        self._x = tf.placeholder(tf.float32, shape=[vec_size], name="model_x")
        self._y = tf.placeholder(tf.float32, shape=[class_size], name="model_y")
        inputs = self._x
