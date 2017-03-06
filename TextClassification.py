import numpy as np
import tensorflow as tf
import sys
import os
import string


WORD_VECTOR = {}
flags = tf.flags
logging = tf.logging
flags.DEFINE_string("vector_file", None, "vector_file")
flags.DEFINE_string("train_data_path", None, "train_data_path")
flags.DEFINE_string("test_data_path", None, "test_data_path")
FLAGS = flags.FLAGS

VEC_SIZE = 300

class Config(object):
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    vec_size = VEC_SIZE
    max_epoch = 6
    max_max_epoch = 59
    keep_prob = 0.5
    lr_decay = 0.8
    class_size = 1  # 1bit. 1 for positive, 0 for negative.

class SoftmaxClassifier(object):
    def __init__(self, is_training=True, config=Config()):
        self._config = config
        self.vec_size = vec_size = config.vec_size
        self.class_size = class_size = config.class_size
        self._x = tf.placeholder(tf.float32, shape=[vec_size], name="model_x")
        self._y = tf.placeholder(tf.float32, shape=[class_size], name="model_y")
        inputs = self._x

        softmax_w = tf.get_variable("softmax_w", [vec_size, class_size])
        softmax_b = tf.get_variable("softmax_b", [class_size])

        y = tf.matmul(inputs, softmax_w) + softmax_b
        self.y = tf.nn.softmax(y)

        self._cost = cost = tf.nn.softmax_cross_entropy_with_logits(y, self._y)

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)  # declare lr

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

        def assign_lr(self, session, lr_value):
            session.run(tf.assign(self._lr, lr_value))

        @property
        def inputs(self):
            return self._x

        @property
        def targets(self):
            return self._y

        @property
        def config(self):
            return self._config

        @property
        def cost(self):
            return self._cost

        @property
        def y(self):
            return self.y

        @property
        def lr(self):
            return self._lr

        @property
        def train_op(self):
            return self._train_op


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
        return WORD_VECTOR[w]
    except KeyError:
        return np.zeros(VEC_SIZE)
    except:
        print "Unexpected error:", sys.exc_info()[0]
        return np.zeros(VEC_SIZE)


def get_text_avg_vector(text):
    # TODO: get the average vectors of the words in the input text
    # Remove Commas?
    vector_sum = np.zeros_like(get_vector("the"))
    text = text.translate(None, string.punctuation)
    for n, word in enumerate(text.split(" ")):
        word = word.strip()
        print "\n"+word+"\n"
        print get_vector(word).shape
        vector_sum += get_vector(word)
    return vector_sum / float(n)


def train_review(text, label):
    input_vector = get_text_avg_vector(text)


def train_pn_files(path, label, session):
    file_list = os.listdir(path)
    for filename in file_list:
        with open(filename) as f:
            text = f.read()
        train_review(text, label)

def train():
    if not FLAGS.train_data_path:
        raise ValueError("Must set --train_data_path")
    if not FLAGS.test_data_path:
        raise ValueError("Must set --test_data_path")
    config = Config()
    train_data_p = FLAGS.train_data_path + "/pos"
    train_data_n = FLAGS.train_data_path + "/neg"
    test_data_p = FLAGS.test_data_path + "/pos"
    test_data_n = FLAGS.test_data_path + "/neg"
    # TODO: init session and train then test


