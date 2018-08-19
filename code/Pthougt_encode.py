import tensorflow as tf
import numpy as np
import gensim
import os
import time
from aa import helper as hp
from aa.utils import *


class Pthought_vec:

    def __init__(self, num_hidden, num_layers, embedding_size, bi_direction=True):
        self.num_hidden = num_hidden
        self.word_emb_dim = 300
        self.num_layers = num_layers
        self.bi_direction = bi_direction

        self._create_placeholders()
        print("placeholders created !")
        self._create_encoder()
        print("encoder created !")

    def set_glove_path(self, glove_path):
        self.glove_path = glove_path

    def get_word_dict(self, sentences, tokenize=True):
        # create vocab of words
        word_dict = {}
        if tokenize:
            from nltk.tokenize import word_tokenize
        sentences = [s.split() if not tokenize else word_tokenize(s)
                     for s in sentences]
        for sent in sentences:
            for word in sent:
                if word not in word_dict:
                    word_dict[word] = ''
        word_dict['<s>'] = ''
        word_dict['</s>'] = ''
        word_dict['unk'] = ''
        return word_dict

    def get_glove(self, word_dict):
        assert hasattr(self, 'glove_path'), \
               'warning : you need to set_glove_path(glove_path)'
        # create word_vec with glove vectors
        word_vec = {}
        with open(self.glove_path,'r',encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in word_dict:
                    word_vec[word] = np.fromstring(vec, sep=' ')
        print('Found {0}(/{1}) words with glove vectors'.format(
                    len(word_vec), len(word_dict)))
        return word_vec

    def build_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'glove_path'), 'warning : you need \
                                             to set_glove_path(glove_path)'
        word_dict = self.get_word_dict(sentences, tokenize)
        self.word_vec = self.get_glove(word_dict)
        print('Vocab size : {0}'.format(len(self.word_vec)))

    def get_batch(self, batch, lengths):
        # sent in batch in decreasing order of lengths
        # batch: (bsize, max_len, word_dim)
        embed = np.zeros((np.max(lengths), len(batch), self.word_emb_dim))

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]

        return embed

    def prepare_samples(self, sentences, tokenize=True, verbose=True):
        if tokenize:
            from nltk.tokenize import word_tokenize
        #sentences = [['<s>'] + s.split() + ['</s>'] if not tokenize else
        #             ['<s>']+word_tokenize(s)+['</s>'] for s in sentences]
        sentences = [s.split() if not tokenize else
                     word_tokenize(s) for s in sentences]
        n_w = np.sum([len(x) for x in sentences])

        # filters words without glove vectors
        for i in range(len(sentences)):
            #s_f = [word for word in sentences[i] if word in self.word_vec]
            s_f = []
            for k,w in enumerate(sentences[i]):
                if w in self.word_vec:
                    s_f.append(w)
                else:
                    s_f.append('unk')
            if not s_f:
                import warnings
                warnings.warn('No words in "{0}" (idx={1}) have glove vectors. \
                               Replacing by "</s>"..'.format(sentences[i], i))
                s_f = ['</s>']
            sentences[i] = s_f

        lengths = np.array([len(s) for s in sentences])
        n_wk = np.sum(lengths)
        if verbose:
            print('Nb words kept : {0}/{1} ({2} %)'.format(
                        n_wk, n_w, round((100.0 * n_wk) / n_w, 2)))

        return np.array(sentences), lengths

###################
    def _create_placeholders(self):
        with tf.variable_scope('placeholders'):
            # encoder
            self.encoder_sequence_len = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_sequence_len')
            self.encoder_inputs_embedded = tf.placeholder(shape=(None, None, self.word_emb_dim), dtype=tf.float32, name='encoder_inputs_embed')

    def _gru_cell(self, num_hidden):
        return tf.contrib.rnn.GRUCell(num_hidden)

    def _create_encoder(self):
        with tf.variable_scope('encoder'):
            if self.bi_direction == True:
                cell_fw = self._gru_cell(self.num_hidden)
                cell_bw = self._gru_cell(self.num_hidden)

                if self.num_layers > 1:
                    cell_fw = tf.nn.rnn_cell.MultiRNNCell(
                        [self._gru_cell(self.num_hidden) for _ in range(self.num_layers)])
                    cell_bw = tf.nn.rnn_cell.MultiRNNCell(
                        [self._gru_cell(self.num_hidden) for _ in range(self.num_layers)])

                encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs=self.encoder_inputs_embedded,
                                                                                sequence_length=self.encoder_sequence_len,
                                                                                dtype=tf.float32, time_major=True, )
                if self.num_layers == 1:
                    encoder_state = encoder_state
                    self.encoder_state_concat = tf.concat(encoder_state, axis=1)
                else:
                    assert isinstance(encoder_state, tuple)
                    encoder_state_fw = encoder_state[0][-1]
                    encoder_state_bw = encoder_state[1][-1]
                    self.encoder_state_concat = tf.concat((encoder_state_fw, encoder_state_bw), 1)
            else:
                if self.num_layers > 1:
                    cell_fw = tf.nn.rnn_cell.MultiRNNCell(
                        [self._gru_cell(self.num_hidden) for _ in range(self.num_layers)])

                encoder_output, encoder_state = tf.nn.dynamic_rnn(cell_fw, inputs=self.encoder_inputs_embedded,
                                                                  sequence_length=self.encoder_sequence_len,
                                                                  dtype=tf.float32, time_major=True, )
                if self.num_layers == 1:
                    self.encoder_state_concat = encoder_state
                else:
                    assert isinstance(encoder_state, tuple)
                    self.encoder_state_concat = tf.concat(encoder_state, axis=1)

    def enc_saver(self):
        # Encoder
        self.saver_enc = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder'))

    def load_enc(self, model_path, model_name, sess):
        restorename = model_path + "/" + model_name
        self.saver_enc.restore(sess, restorename)

    def encode(self,sess, sentences, bsize=128,tokenize=True, verbose=False):
        sentences, lengths = self.prepare_samples(sentences, tokenize, verbose)
        embeddings = []
        for stidx in range(0, len(sentences), bsize):
            batch = self.get_batch(sentences[stidx:stidx + bsize],lengths[stidx:stidx+bsize])
            batch = sess.run(self.encoder_state_concat,feed_dict={self.encoder_inputs_embedded:batch, self.encoder_sequence_len:lengths[stidx:stidx+bsize]})
            embeddings.append(batch)
        embeddings = np.vstack(embeddings)
        return embeddings


