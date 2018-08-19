import tensorflow as tf
import numpy as np
import gensim
import os
import time
from aa import helper as hp
from aa.utils import *


class Pthought_vec:

    def __init__(self, num_hidden, num_layers, bi_direction, softmax_sampling_size, glove_path, corpus, lamb):
        self.num_hidden = num_hidden
        self.word_emb_dim = 300
        self.num_layers = num_layers
        self.softmax_sampling_size = softmax_sampling_size
        self.para_weight = lamb
        self.bi_direction=bi_direction
        self.xavier = tf.contrib.layers.xavier_initializer()

        self.set_glove_path(glove_path)
        self.build_vocab(corpus, tokenize=True)
        self.ind = {k:i for i,k in enumerate(self.word_vec.keys())}

        self._create_placeholders()
        print("placeholders created !")
        self._create_embedding_matrix()
        print("embedding matrix created !")
        self._create_network()
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
        with open(self.glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in word_dict:
                    word_vec[word] = np.fromstring(vec, sep=' ')
        print('Found {0}(/{1}) words with glove vectors'.format(
            len(word_vec), len(word_dict)))


        self.vocab_size = len(word_vec)
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
        # sentences = [['<s>'] + s.split() + ['</s>'] if not tokenize else
        #             ['<s>']+word_tokenize(s)+['</s>'] for s in sentences]
        sentences = [s.split() if not tokenize else
                     word_tokenize(s) for s in sentences]
        n_w = np.sum([len(x) for x in sentences])

        # filters words without glove vectors
        for i in range(len(sentences)):
            #s_f = [word for word in sentences[i] if word in self.word_vec]
            s_f = []
            for k, w in enumerate(sentences[i]):
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
            self.encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
            self.encoder_sequence_len = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_sequence_len')

            # decoder_auto
            self.auto_decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='auto_decoder_targets')
            self.auto_decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='auto_decoder_inputs')
            self.auto_decoder_sequence_len = tf.placeholder(shape=(None,), dtype=tf.int32,
                                                            name='auto_decoder_sequence_len')

            # decoder_paraphrase
            self.para_decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='para_decoder_targets')
            self.para_decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='para_decoder_inputs')
            self.para_decoder_sequence_len = tf.placeholder(shape=(None,), dtype=tf.int32,
                                                            name='para_decoder_sequence_len')

            self.lr = tf.placeholder(shape=(), dtype=tf.float32, name='learning_rate')

    def _create_embedding_matrix(self):
        # Embedding matrix
        # denote lookup table as placeholder
        self.embeddings = tf.constant(np.array([x for x in list(self.word_vec.values())],dtype=np.float32))
        # self.embeddings = tf.get_variable('embedding',shape=[self.vocab_size,self.embedding_size],
        #                                  initializer=self.xavier,dtype=tf.float32)

        self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.encoder_inputs)
        self.auto_decoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.auto_decoder_inputs)
        self.para_decoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.para_decoder_inputs)

    def _gru_cell(self, num_hidden):
        return tf.contrib.rnn.GRUCell(num_hidden)

    def _LSTM_cell(self, num_hidden):
        return lambda x: tf.nn.rnn_cell.BasicLSTMCell(x, state_is_tuple=True)

    def _create_encoder(self, embedded):
        with tf.variable_scope('encoder'):
            if self.bi_direction == True:
                cell_fw = self._gru_cell(self.num_hidden)
                cell_bw = self._gru_cell(self.num_hidden)

                if self.num_layers > 1:
                    cell_fw = tf.nn.rnn_cell.MultiRNNCell(
                        [self._gru_cell(self.num_hidden) for _ in range(self.num_layers)])
                    cell_bw = tf.nn.rnn_cell.MultiRNNCell(
                        [self._gru_cell(self.num_hidden) for _ in range(self.num_layers)])

                encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs=embedded,
                                                                                sequence_length=self.encoder_sequence_len,
                                                                                dtype=tf.float32, time_major=True, )
                if self.num_layers == 1:
                    encoder_state = encoder_state
                    encoder_state_concat = tf.concat(encoder_state, axis=1)
                else:
                    assert isinstance(encoder_state, tuple)
                    encoder_state_fw = encoder_state[0][-1]
                    encoder_state_bw = encoder_state[1][-1]
                    encoder_state_concat = tf.concat((encoder_state_fw, encoder_state_bw), 1)
            else:
                if self.num_layers > 1:
                    cell_fw = tf.nn.rnn_cell.MultiRNNCell(
                        [self._gru_cell(self.num_hidden) for _ in range(self.num_layers)])

                encoder_output, encoder_state = tf.nn.dynamic_rnn(cell_fw, inputs=embedded,
                                                                  sequence_length=self.encoder_sequence_len,
                                                                  dtype=tf.float32, time_major=True, )
                if self.num_layers == 1:
                    encoder_state_concat = encoder_state
                else:
                    assert isinstance(encoder_state, tuple)
                    encoder_state_concat = tf.concat(encoder_state, axis=1)

        return encoder_state_concat

    def _create_decoder(self, scope_name, encoder_state, decoder_embed, decoder_len):
        with tf.variable_scope(scope_name):
            cell = self._gru_cell(self.num_hidden * 2)

            decoder_outputs, _ = tf.nn.dynamic_rnn(cell, decoder_embed, sequence_length=decoder_len,
                                                   initial_state=encoder_state, dtype=tf.float32,
                                                   time_major=True, )
            # variables for softmax
            w = tf.get_variable("proj_w", shape=[self.vocab_size, self.num_hidden * 2],
                                initializer=self.xavier)
            b = tf.get_variable("proj_b", shape=[self.vocab_size], initializer=self.xavier)

        return decoder_outputs, cell, w, b

    def _create_network(self):
        # encoder
        self.encoder_state = self._create_encoder(self.encoder_inputs_embedded)

        # paraphrase_decoder
        para_decoder_outputs, _, para_w, para_b = self._create_decoder("para_decoder", self.encoder_state,
                                                                       self.para_decoder_inputs_embedded,
                                                                       self.para_decoder_sequence_len)
        # Auto_decoder
        auto_decoder_outputs, self.AE_cell, self.auto_w, self.auto_b = self._create_decoder("auto_decoder",
                                                                                            self.encoder_state,
                                                                                            self.auto_decoder_inputs_embedded,
                                                                                            self.auto_decoder_sequence_len)

        def cal_sampled_loss(w, b, target, input):
            cross_entropy_loss = tf.nn.sampled_softmax_loss(weights=w, biases=b,
                                                            labels = tf.reshape(target, [-1, 1]),
                                                            inputs = tf.reshape(input, [-1, self.num_hidden * 2]),
                                                            num_sampled= self.softmax_sampling_size,
                                                            num_classes= self.vocab_size,
                                                            num_true=1)
            loss = tf.reduce_mean(cross_entropy_loss)
            return loss, cross_entropy_loss

        self.para_loss, self.para_cel = cal_sampled_loss(para_w, para_b, self.para_decoder_targets,
                                                         para_decoder_outputs)
        self.auto_loss, self.auto_cel = cal_sampled_loss(self.auto_w, self.auto_b, self.auto_decoder_targets,
                                                         auto_decoder_outputs)

        self.loss = self.para_weight * self.para_loss + self.auto_loss
        # self.loss = self.para_loss + self.auto_loss + self.para_weight * tf.abs(self.para_loss - self.auto_loss)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def log_and_saver(self, log_path, model_path, sess):
        # log
        loss_sum = tf.summary.scalar("Loss", self.loss)
        self.summary = tf.summary.merge_all()

        self.writer_tr = tf.summary.FileWriter(log_path + "/train", sess.graph)
        self.writer_test = tf.summary.FileWriter(log_path + "/test", sess.graph)

        # saver
        self.dir = os.path.dirname(os.path.realpath(model_path))

    def saver(self):
        self.all_saver = tf.train.Saver()

    def next_feed(self, input, para, lr):
        self.encoder_inputs_, self.en_seq_len_ = hp.batch(input)
        # self.encoder_inputs_, self.en_seq_len_ = hp.batch([[self.ST] + (sequence) + [self.EOS] for sequence in input])

        self.auto_targets_, self.auto_seq_len_ = hp.batch([(sequence) + [self.ind['</s>']] for sequence in input])
        self.auto_inputs_, _ = hp.batch([[self.ind['</s>']] + (sequence) for sequence in input])

        self.para_targets_, self.para_seq_len_ = hp.batch([(sequence) + [self.ind['</s>']] for sequence in para])
        self.para_inputs_, _ = hp.batch([[self.ind['</s>']] + (sequence) for sequence in para])

        return {
            self.encoder_inputs: self.encoder_inputs_,
            self.encoder_sequence_len: self.en_seq_len_,

            self.auto_decoder_inputs: self.auto_inputs_,
            self.auto_decoder_targets: self.auto_targets_,
            self.auto_decoder_sequence_len: self.auto_seq_len_,

            self.para_decoder_inputs: self.para_inputs_,
            self.para_decoder_targets: self.para_targets_,
            self.para_decoder_sequence_len: self.para_seq_len_,

            self.lr: lr
        }

    def variable_initialize(self, sess):
        sess.run(tf.global_variables_initializer())

    ## model trainer
    def train(self, input_train, input_val, target_train, target_val, batch_size, n_epoch, init_lr, sess, tokenize = True):
        print("Start train !!!!!!!")
        from nltk.tokenize import word_tokenize
        count_t = time.time()
        for i in range(n_epoch):
            for start in range(0, len(input_train), batch_size):
                batch_time = time.time()
                global_iter = i * int(len(input_train) / batch_size) + int(start / batch_size + 1)
                lr = init_lr * pow(0.5, i)
                
                sentences_input = [s.split() if not tokenize else word_tokenize(s) for s in input_train[start:start + batch_size]]
                input_train_idx = []
                for _, v in enumerate(sentences_input):
                    input_train_idx.append([self.ind[s] if s in self.word_vec.keys() else self.ind['unk'] for s in v])

                sentences_target = [s.split() if not tokenize else word_tokenize(s) for s in target_train[start:start + batch_size]]
                target_train_idx = []
                for _, v in enumerate(sentences_target):
                    target_train_idx.append([self.ind[s] if s in self.word_vec.keys() else self.ind['unk'] for s in v])


                ## training
                fd = self.next_feed(input_train_idx, target_train_idx, lr)
                s_tr, _, l_tr, l_auto, l_para = sess.run(
                    [self.summary, self.train_op, self.loss, self.auto_loss, self.para_loss], feed_dict=fd)
                self.writer_tr.add_summary(s_tr, global_iter)

                # validation
                tst_idx = np.arange(len(target_val))
                np.random.shuffle(tst_idx)
                tst_idx = tst_idx[0:batch_size * 3]

                sentences_input = [s.split() if not tokenize else word_tokenize(s) for s in np.take(input_val, tst_idx, 0)]
                input_val_idx = []
                for _, v in enumerate(sentences_input):
                    input_val_idx.append([self.ind[s] if s in self.word_vec.keys() else self.ind['unk'] for s in v])

                sentences_target = [s.split() if not tokenize else word_tokenize(s) for s in np.take(target_val, tst_idx, 0)]
                target_val_idx = []
                for _, v in enumerate(sentences_target):
                    target_val_idx.append([self.ind[s] if s in self.word_vec.keys() else self.ind['unk'] for s in v])


                fd_tst = self.next_feed(input_val_idx, target_val_idx, lr)

                s_tst, l_tst, l_auto_tst, l_para_tst = sess.run([self.summary, self.loss, self.auto_loss, self.para_loss], feed_dict=fd_tst)
                self.writer_test.add_summary(s_tst, global_iter)

                if start == 0 or int(start / batch_size + 1) % 50 == 0:
                    print("Epoch", i + 1, "Iter", int(start / batch_size + 1), " Training Loss:", l_tr, "Test loss : ",
                          l_tst, "Time : ", time.time() - batch_time)
                    print("Training Auto Loss:", l_auto, "Training para Loss:", l_para, "Test Auto Loss:", l_auto_tst,
                          "Test para Loss:", l_para_tst)

            savename = self.dir + "net-" + str(i + 1) + ".ckpt"
            self.all_saver.save(sess=sess, save_path=savename)

            # if (i + 1) % 5 == 0:
            #    savename = self.dir + "net-" + str(i + 1) + ".ckpt"
            #    self.all_saver.save(sess=sess, save_path=savename)

            print("epoch : ", i + 1, "loss : ", l_tr, "Test loss : ", l_tst)

        print("Running Time : ", time.time() - count_t)
        print("Training Finished!!!")

    def load_model(self, model_path, model_name, sess):
        restorename = model_path + "/" + model_name
        self.all_saver.restore(sess, restorename)

    def AE_saver(self):
        # Encoder
        self.saver_enc = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder'))

        # Decoder
        self.saver_dec = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='auto_decoder'))

        self.weight_saver = tf.train.Saver({"auto_decoder/proj_w": self.auto_w})
        self.bias_saver = tf.train.Saver({"auto_decoder/proj_b": self.auto_b})

    def load_AE(self, model_path, model_name, sess):
        restorename = model_path + "/" + model_name
        self.saver_enc.restore(sess, restorename)
        self.saver_dec.restore(sess, restorename)

        self.weight_saver.restore(sess, restorename)
        self.bias_saver.restore(sess, restorename)

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


