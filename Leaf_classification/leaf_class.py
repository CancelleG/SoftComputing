#采用了一维卷积神经网络实现了叶子分类，去除了pooling层
#后期工作：弄清楚StraitfiedShuffleSplit的作用
####################################
#A=[[1,2,3],[4,5,6],[7,8,9]]
# B=[1,2]
# A = np.array(A)
# B = np.array(B)
# A[B]
# Out[29]:
# array([[4, 5, 6],
#        [7, 8, 9]])


import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit
import os

def get_data():
    train = pd.read_csv('./data/train.csv')
    test = pd.read_csv('./data/test.csv')
    #由于数据集中的label为英文，为了方便运行，采用label_encoder进行处理
    label_encoder = LabelEncoder().fit(train.species)
    labels = label_encoder.transform(train.species)     #labels（0-99），可在classes中查找对应种类
    classes = list(label_encoder.classes_)
    #剔除训练集和测试集中的不必要行和列
    train = train.drop(['species', 'id'],axis=1)
    test = test.drop('id', axis=1)
    return train, labels, test, classes

def main():

    X_width = len(X_train[0])
    label_width = len(y_train[0])
    sess = tf.InteractiveSession()
    x = tf.placeholder("float", shape=[None, X_width])
    y_ = tf.placeholder("float", shape=[None, label_width])
    sess.run(tf.initialize_all_variables())
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    def conv1d(x, W):
        return tf.nn.conv1d(x, W, stride=1, padding='SAME')
    def max_pool_1x2(x):        #x是字母，不是×
        return tf.nn.pool(x, window_shape=[2], strides=[2], padding='SAME', pooling_type='MAX')


    W_conv1 = weight_variable([5,1,32])       #前两维为patch的大小，接着是输入的通道数目，最后是输出的通道数目
    b_conv1 = bias_variable([32])
    x_data = tf.reshape(x, [-1, X_width, 1])       #2、3维为图片的大小（宽和高），4维为通道数，
                                                ############第一维是什么
    h_conv1 = tf.nn.relu(conv1d(x_data, W_conv1) + b_conv1)

    #若要使用pooling层，则再修改
    #tf.nn.pool(x, window_shape=[2], strides=[2], padding='SAME', pooling_type='MAX')
    #h_conv2 = tf.nn.relu(conv1d(h_pool1, W_conv2) + b_conv2)
    #W_fc1 = weight_variable([X_width*32, 1024]) ##############
    # h_pool1 = max_pool_1x2(h_conv1)

    W_conv2 = weight_variable([5,32,64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv1d(h_conv1, W_conv2) + b_conv2)
    # h_pool2 = max_pool_1x2(h_conv2)

    W_fc1 = weight_variable([X_width*64, 1024]) ##############
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_conv2, [-1, X_width*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, label_width])
    b_fc2 = bias_variable([label_width])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    sess.run(tf.initialize_all_variables())
    for step in range(80):
        if step%10 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: X_train, y_: y_train, keep_prob: 1.0})
            print("step %d, training accuracy %g" % (step, train_accuracy))
        train_step.run(feed_dict={x: X_train, y_: y_train, keep_prob: 0.5})
    print("test accuracy:%g"%accuracy.eval(feed_dict={x: X_valid, y_: y_valid,
                                                      keep_prob: 1.0}))



if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   #防止系统报错 Allocation of exceeds  10% of system memory
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


    train, labels, test, classes = get_data()
    #将train中值归一化为-1~1之间的值
    train_scaled = StandardScaler().fit_transform(train.values)
    sss = StratifiedShuffleSplit(test_size=0.1, random_state=23)
    for train_index, valid_index in sss.split(train_scaled, labels):
        X_train, X_valid = train_scaled[train_index], train_scaled[valid_index]
        y_train, y_valid = labels[train_index], labels[valid_index]
    OneHot = OneHotEncoder().fit(y_train.reshape(-1, 1))
    y_train = OneHot.transform(y_train.reshape(-1, 1)).toarray()
    y_valid = OneHotEncoder().fit_transform(y_valid.reshape(-1, 1)).toarray()
    main()