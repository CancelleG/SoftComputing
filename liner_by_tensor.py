import tensorflow as tf
import numpy as np

def main():
    x_data = np.float32(np.random.rand(2, 100))
    y_data = np.dot([0.100, 0.200], x_data)
    b = tf.Variable(tf.zeros([1]))
    # W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0)) #np.random
    W = tf.Variable(np.float32(np.random.rand(1,2)))
    y = tf.matmul(W, x_data) + b

    loss = tf.reduce_mean(tf.square(y - y_data))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)        #tf.train.G....minimize(loss)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    for step in range(100):
        sess.run(train)
        if step%20 == 0:
            print("Step%d"%step, sess.run(W), sess.run(b))



if __name__ == '__main__':
    main()
