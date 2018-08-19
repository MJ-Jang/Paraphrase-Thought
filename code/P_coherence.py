from aa.Pthougt_encode import Pthought_vec
import tensorflow as tf
import numpy as np
import pandas as pd

glove_path = "D:/glove.840B.300d.txt"
word_emb_dim = 300

sess = tf.Session()
model = Pthought_vec(1200,2,300,bi_direction=False)
model.enc_saver()

model.load_enc(model_path= './model/two_forward',model_name="Pthought_net.ckpt",sess=sess)

### P-coherence
set_tst = pd.read_csv("./data/2017val.csv")

cap = np.array(set_tst['caption'])

model.set_glove_path(glove_path)
model.build_vocab(cap,tokenize = True)
total = model.encode(sess, cap, tokenize = True)

img_id = set_tst['image_id']
uniq_id = np.unique(img_id)

from sklearn.metrics.pairwise import cosine_similarity as cs
coherence = []
for i in range(len(uniq_id)):
    # inner
    tmp_id = np.where(img_id == uniq_id[i])[0]
    coherence_vec = np.take(total, tmp_id, 0)
    dist_coherence = cs(coherence_vec)
    mean_coherence = (np.sum(dist_coherence) - len(dist_coherence)) / (len(dist_coherence) * len(dist_coherence) - len(dist_coherence))
    coherence.append(mean_coherence)
np.mean(coherence)