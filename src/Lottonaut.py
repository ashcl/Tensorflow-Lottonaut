import tensorflow as tf
import numpy as np

class Lottonaut:
    def __init__(self):
        pass

    def newNormalDist(self):
        normTensor = tf.random_normal([1000], mean=25, stddev=10)
        with tf.Session() as session:
            return normTensor.eval()

    def newUniformDistribution(self, count=10, min=1, max=65):
        uniformTensor = tf.random_uniform([count], minval=min, maxval=max, dtype=tf.float32)
        with tf.Session() as session:
            return uniformTensor.eval()


if (__name__ == "__main__"):
    lottonaut = Lottonaut()
    print(lottonaut.newNormalDist())
