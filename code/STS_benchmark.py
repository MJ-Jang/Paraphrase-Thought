from aa.Pthougt_encode import Pthought_vec
from aa.STS_function import *
import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import csv
import matplotlib.pyplot as plt

glove_path = "D:/glove.840B.300d.txt"
word_emb_dim = 300

sess = tf.Session()
model = Pthought_vec(1200,2,300,bi_direction=False)
model.enc_saver()

model.load_enc(model_path= './model/two_forward',model_name="Pthought_net.ckpt",sess=sess)


## STS
test = pd.read_csv("./data/STS/sts-test.csv",sep="\t",error_bad_lines=False,header=None,quoting=csv.QUOTE_NONE)
tst_A = np.array(test[5])
tst_B = np.array(test[6])
tst_Y = np.array(test[4],dtype=float)

model.set_glove_path(glove_path)
model.build_vocab(tst_A,tokenize = True)
tst_A = model.encode(sess, tst_A, tokenize = True)

model.build_vocab(tst_B,tokenize = True)
tst_B = model.encode(sess,tst_B, tokenize=True)
abs_tst = np.abs(tst_A - tst_B)
dot_tst = np.multiply(tst_A,tst_B)
tst_X = np.c_[abs_tst,dot_tst]

# train
train = pd.read_csv("C:/Users/MJ/Desktop/stsbenchmark/sts-train.csv",sep="\t",error_bad_lines=False,header=None,quoting=csv.QUOTE_NONE)

tr_A = np.array(train[5])
tr_B = np.array(train[6])
tr_Y = np.array(train[4],dtype=float)

model.build_vocab(tr_A,tokenize = True)
tr_A = model.encode(sess, tr_A, tokenize = True)

model.build_vocab(tr_B,tokenize = True)
tr_B = model.encode(sess,tr_B, tokenize=True)

abs = np.abs(tr_A - tr_B)
dot = np.multiply(tr_A,tr_B)
tr_X = np.c_[abs,dot]

dev = pd.read_csv("C:/Users/MJ/Desktop/stsbenchmark/sts-dev.csv",sep="\t",error_bad_lines=False,header=None,quoting=csv.QUOTE_NONE)
dev_A = np.array(dev[5])
dev_B = np.array(dev[6])
dev_Y = np.array(dev[4],dtype=float)

model.build_vocab(dev_A,tokenize = True)
dev_A = model.encode(sess, dev_A, tokenize = True)

model.build_vocab(dev_B,tokenize = True)
dev_B = model.encode(sess,dev_B, tokenize=True)

abs_dev = np.abs(dev_A - dev_B)
dot_dev = np.multiply(dev_A,dev_B)
dev_X = np.c_[abs_dev,dot_dev]

yhat_tst = Logistic_Reg(tr_X,tr_Y,dev_X,dev_Y,tst_X,tst_Y)
print("Pearson cor:",pearsonr(yhat_tst,tst_Y)[0], "Spearman cor:",spearmanr(yhat_tst,tst_Y)[0])

# Scatter plot
df = pd.DataFrame( np.c_[yhat_tst,tst_Y], columns=['Pred',"Truth"])
ax = df.plot.hexbin(x='Pred',y="Truth",gridsize=20)
ax.text(1, 4.8, r'pearsonr = %.3f, spearmanr = %.3f' % (pearsonr(yhat_tst,tst_Y)[0],spearmanr(yhat_tst,tst_Y)[0]))
ax.set_facecolor('white')
plt.show()