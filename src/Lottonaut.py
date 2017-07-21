import tensorflow as tf
import numpy as np

class Lottonaut:
    def __init__(self):
        pass
    def newNormalDist(selfself):
        norm = tf.random_normal([1000], mean=25, stddev=10)
        with tf.Session() as session:
            return norm.eval()


if (__name__ == "__main__"):
    lottonaut = Lottonaut()
    print(lottonaut.newNormalDist())