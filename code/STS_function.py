import numpy as np
import tensorflow as tf
from scipy.stats import pearsonr, spearmanr

def encode_labels(labels, nclass=5):
    """
    Label encoding from Tree LSTM paper (Tai, Socher, Manning)
    """
    Y = np.zeros((len(labels), nclass)).astype('float32')
    for j, y in enumerate(labels):
        for i in range(nclass):
            if i + 1 == np.floor(y) + 1:
                Y[j, i] = y - np.floor(y)
            if i + 1 == np.floor(y):
                Y[j, i] = np.floor(y) - y + 1
    return Y

def Logistic_Reg(tr_X,tr_Y,dev_X,dev_Y,tst_X,tst_Y):
    tr_Y_t = encode_labels(tr_Y)
    tst_Y_t = encode_labels(tst_Y)
    dev_Y_t = encode_labels(dev_Y)

    tf.reset_default_graph()
    X = tf.placeholder(tf.float64, shape=[None, 4800], name="train_input")
    Y = tf.placeholder(tf.float64, shape=[None, 5], name="train_Y")

    # params
    W = tf.get_variable(name="W", shape=[4800, 5], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
    b = tf.get_variable(name="b", shape=[5], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)

    output = tf.matmul(X, W) + b
    prob = tf.nn.softmax(output)

    loss = tf.losses.mean_squared_error(Y, prob)
    train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

    sess = tf.Session()

    r = np.arange(1, 6)
    batch_size = 64
    bestpr = -1
    early_stop_count = 0
    stop_train = False
    max_epoch = 500
    early_stop = True
    i = 0

    sess.run(tf.global_variables_initializer())
    while not stop_train and i <= max_epoch:
        for start in range(0, len(tr_X), batch_size):
            _, l_tr, p_tr = sess.run([train_op, loss, prob], feed_dict={X: tr_X[start:start + batch_size],
                                                                        Y: tr_Y_t[start:start + batch_size]})

            yhat_tr = np.dot(p_tr, r)
            cor_tr = pearsonr(yhat_tr, tr_Y[start:start + batch_size])

        l_dev, p_dev = sess.run([loss, prob], feed_dict={X: dev_X, Y: dev_Y_t})
        yhat_dev = np.dot(p_dev, r)
        cor_dev = pearsonr(yhat_dev, dev_Y)

        p_tst = sess.run(prob, feed_dict={X: tst_X, Y: tst_Y_t})
        yhat_tst = np.dot(p_tst, r)
        cor_tst = pearsonr(yhat_tst, tst_Y)
        print("### Epoch :", i, "Loss tr :", l_tr, "Loss dev :", l_dev, "Corr tr :", cor_tr[0], "Cor dev :", cor_dev[0],
              "Cor tst :", cor_tst[0])

        if cor_dev[0] > bestpr:
            bestpr = cor_dev[0]

        elif early_stop:
            if early_stop_count >= 3:
                stop_train = True
            early_stop_count += 1
        i += 1

    # min-max scaleing, correlation does not change
    return 5*(yhat_tst -min(yhat_tst))/(max(yhat_tst)-min(yhat_tst))