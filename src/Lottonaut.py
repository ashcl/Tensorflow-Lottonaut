import tensorflow as tf

class Lottonaut:
    def __init__(self):
        self.state = "Powerball"

    def newNormalDist(self):
        normTensor = tf.random_normal([1000], mean=25, stddev=10)
        with tf.Session() as session:
            return normTensor.eval()

    def newUniformDistribution(self, count, min=1, max=65):
        uniformTensor = tf.random_uniform([count], minval=min, maxval=max, dtype=tf.float32)
        with tf.Session() as session:
            return uniformTensor.eval()

    def guessWhiteBalls(self, min, max):
        balls = self.newUniformDistribution(5, min, max)
        for idx, ball in enumerate(balls):
            balls[idx] = int(round(ball))
        return balls

if (__name__ == "__main__"):
    print("Performing Lottonaut systems check. Stand by...")
