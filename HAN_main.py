import argparse
import re

import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
import gensim
from keras.utils.np_utils import to_categorical
from sklearn.metrics import classification_report
from tensorflow.python.ops.rnn_cell_impl import GRUCell
from tqdm import tqdm
from nltk import tokenize
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from helper import *
from HAN_model import HAN_Model
from HAN_utils import batch_iter
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class HAN(object):

    def __init__(self, params, epochs, batch_size, device, lr, text_list, vocab, fake_label):
        self.p = params
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.lr = lr
        self.text_list = text_list
        self.vocab = vocab
        self.fake_label = fake_label
        assert len(self.text_list) == len(self.fake_label)

        self.MAX_SENT_LENGTH = 200  # max in doc is 966
        self.MAX_SENTS = 50  # max in doc is 282
        # self.MAX_NB_WORDS = 10000
        self.VALIDATION_SPLIT = 0.2

        fname1, fname2, fname3 = './file/HAN_data.pkl', './file/HAN_word_length.pkl', './file/HAN_sent_lengt.pkl'
        if not checkFile(fname1) or not checkFile(fname2):
            data = np.zeros((len(self.text_list), self.MAX_SENTS, self.MAX_SENT_LENGTH), dtype='int32')
            word_length = np.zeros((len(self.text_list), self.MAX_SENTS), dtype='int32')
            sent_length = np.zeros((len(self.text_list)), dtype='int32')

            for i, sentences in enumerate(self.text_list):
                # sent_len is number of sentences in a doc
                sent_len = 0
                # print('i:', i, 'sentences:', sentences)
                for j, sent in enumerate(sentences):
                    if j < self.MAX_SENTS:
                        sent_len += 1
                        # print('j:', j, 'sent:', sent)
                        wordTokens = nltk.word_tokenize(sent)
                        # print('wordTokens:', type(wordTokens), wordTokens)

                        # k is number of words in a sentence
                        k = 0
                        for _, word in enumerate(wordTokens):
                            # print('_:', _, 'word:', word)
                            if k < self.MAX_SENT_LENGTH:
                                data[i, j, k] = self.vocab.index(word)
                                k = k + 1

                        word_length[i, j] = k
                sent_length[i] = sent_len
            pickle.dump(data, open(fname1, 'wb'))
            pickle.dump(word_length, open(fname2, 'wb'))
            pickle.dump(sent_length, open(fname3, 'wb'))
        else:
            data = pickle.load(open(fname1, 'rb'))
            word_length = pickle.load(open(fname2, 'rb'))
            sent_length = pickle.load(open(fname3, 'rb'))
        # print(data[:5])
        # print(sent_length[:5])
        # print(word_length[:5])

        print('max(sent_length):', max(sent_length))
        print(max(np.reshape(word_length, [-1])))
        # exit(0)
        print('data.shape:', data.shape)

        vocab = self.vocab

        f = open('./file/vocab.txt', 'w')
        for i in range(len(vocab)):
            f.write(str(vocab[i]))
            f.write('\n')
        f.close()

        print('Total %s unique tokens.' % len(vocab))
        self.vocab_size = len(vocab)

        labels = []
        for idx in range(len(self.text_list)):
            labels.append(self.fake_label[idx])

        print('generate init word2vec dict embeddings')
        print('use pre-trained vectors:', self.p.embed_loc)
        model = gensim.models.KeyedVectors.load_word2vec_format(self.p.embed_loc, binary=False)
        self.pre_train_context_embedding = look_up_Embeddings(model, vocab, self.p.embed_dims)
        print('self.pre_train_context_embedding:', type(self.pre_train_context_embedding), len(self.pre_train_context_embedding))

        labels = to_categorical(np.asarray(labels))
        print('Shape of data tensor:', data.shape)
        print('Shape of label tensor:', labels.shape)
        self.classes = labels.shape[-1]

        indices = np.arange(data.shape[0])
        print('indices:', type(indices), indices)
        np.random.shuffle(indices)
        data = data[indices]
        labels = labels[indices]
        sent_length = sent_length[indices]
        word_length = word_length[indices]
        nb_validation_samples = int(self.VALIDATION_SPLIT * data.shape[0])

        # self.X_train = data[:-nb_validation_samples]
        # self.y_train = labels[:-nb_validation_samples]
        # self.sent_length_train = sent_length[:-nb_validation_samples]
        # self.word_length_train = word_length[-nb_validation_samples:]
        self.X_train = data
        self.y_train = labels
        self.sent_length_train = sent_length
        self.word_length_train = word_length

        # self.X_val = data[-nb_validation_samples:]
        # self.y_val = labels[-nb_validation_samples:]
        # self.sent_length_val = sent_length[-nb_validation_samples:]
        # self.word_length_val = word_length[-nb_validation_samples:]
        self.X_val = data
        self.y_val = labels
        self.sent_length_val = sent_length
        self.word_length_val = word_length

        print('x_train', self.X_train.shape)
        print('y_train', self.y_train.shape)
        print('x_val', self.X_val.shape)
        print('y_val', self.y_val.shape)

        print('Number of positive and negative reviews in training and validation set')
        print(self.y_train.sum(axis=0))
        print(self.y_val.sum(axis=0))

        # building Hierachical Attention network
        self.allow_soft_placement = True
        self.log_device_placement = False

        self.output_embedding, self.prediction_result = self.train()


    def remove_symbols(self, text):
        del_str = string.punctuation + string.digits
        # _punctuation = """!"#$%&'()*+,-/:;<=>?@[\]^_`{|}~"""
        # del_str = _punctuation + string.digits
        replace = ' ' * len(del_str)
        tran_tab = str.maketrans(del_str, replace)
        text = text.translate(tran_tab)
        return text

    def clean_text(self, text):
        # text = text.lower()
        only_word = self.remove_symbols(text)
        sub = nltk.word_tokenize(only_word)
        without_stopwords = [w for w in sub if not w in stopwords.words('english')]
        # stemmer = PorterStemmer()
        # cleaned_text_list = [stemmer.stem(s) for s in without_stopwords]
        cleaned_text = str()
        # for word in cleaned_text_list:
        for word in without_stopwords:
            cleaned_text = cleaned_text + str(word) + ' '
        return cleaned_text

    def validate(self, epoch, model, sess, X_val, sent_length_val, word_length_val, y_val, batch_size, is_training=True):

        batches = batch_iter(list(zip(X_val, sent_length_val, word_length_val, y_val)),
                             batch_size)

        l = []
        a = []
        all_preds = []
        for i, batch in tqdm(enumerate(batches)):
            X_batch, sent_len_batch, word_lenght_batch, y_batch = zip(
                *batch)
            # print('batch_hist_v', len(batch_utt_v))
            feed_dict = {
                model.inputs: X_batch,
                model.sentence_lengths: sent_len_batch,
                model.word_lengths: word_lenght_batch,
                model.labels: y_batch,
                model.is_training: is_training,
            }

            step, loss, accuracy, predictions, sentence_level_output = sess.run(
                [model.global_step, model.loss, model.accuracy, model.prediction, model.sentence_level_output],
                feed_dict)

            l.append(loss)
            a.append(accuracy)
            all_preds.append(predictions)

        all_preds = np.concatenate(all_preds, axis=0)
        acc = np.average(a)
        print("EVAL: Epoch {}:, loss {:g}, Accuracy {:g}".format(epoch, np.average(l), acc))
        precision = sklearn.metrics.precision_score(np.argmax(y_val, 1), all_preds, average='macro')
        recall = sklearn.metrics.recall_score(np.argmax(y_val, 1), all_preds, average='macro')
        F1 = sklearn.metrics.f1_score(np.argmax(y_val, 1), all_preds, average='macro')
        print("\tmacro-Precision: {:g} ; Recall: {:g} ; F1 {:g}".format(precision, recall, F1))

        precision = sklearn.metrics.precision_score(np.argmax(y_val, 1), all_preds, average='micro')
        recall = sklearn.metrics.recall_score(np.argmax(y_val, 1), all_preds, average='micro')
        F1 = sklearn.metrics.f1_score(np.argmax(y_val, 1), all_preds, average='micro')
        print("\tmicro-Precision: {:g} ; Recall: {:g} ; F1 {:g}".format(precision, recall, F1))

        precision = sklearn.metrics.precision_score(np.argmax(y_val, 1), all_preds, average='weighted')
        recall = sklearn.metrics.recall_score(np.argmax(y_val, 1), all_preds, average='weighted')
        F1 = sklearn.metrics.f1_score(np.argmax(y_val, 1), all_preds, average='weighted')
        print("\tweighted-Precision: {:g} ; Recall: {:g} ; F1 {:g}".format(precision, recall, F1))
        report = classification_report(np.argmax(y_val, 1), all_preds)
        return acc, report

    def train(self):
        session_conf = tf.ConfigProto(
            # device_count={'GPU': gpu_count},
            allow_soft_placement=self.allow_soft_placement,
            log_device_placement=self.log_device_placement,
            gpu_options=tf.GPUOptions(allow_growth=True))
        # Training
        # ==================================================
        best_acc = 0
        best_epoch = 0
        best_report = ''
        gpu_device = 1
        with tf.device('/device:GPU:%d' % gpu_device):
            print('Using GPU - ', '/device:GPU:%d' % gpu_device)
            with tf.Graph().as_default():
                sess = tf.Session(config=session_conf)
                with sess.as_default():
                    seed = 1227

                    kernel_init = tf.glorot_uniform_initializer(seed=seed, dtype=tf.float32)
                    bias_init = tf.zeros_initializer()
                    word_cell = GRUCell(50, name='gru', activation=tf.nn.tanh,
                                        kernel_initializer=kernel_init, bias_initializer=bias_init)
                    sent_cell = GRUCell(50, name='gru', activation=tf.nn.tanh,
                                        kernel_initializer=kernel_init, bias_initializer=bias_init)

                    model = HAN_Model(
                        vocab_size=self.vocab_size,
                        embedding_size=300,
                        classes=self.classes,
                        word_cell=word_cell,
                        sentence_cell=sent_cell,
                        word_output_size=300,
                        sentence_output_size=300,
                        pre_train_embedding=self.pre_train_context_embedding,
                        dropout_keep_proba=0.5,
                        learning_rate=self.lr,
                        device=self.device,
                        scope='HANModel'
                    )
                    sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
                    # tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

                    print("\nEvaluation before training:")
                    # Evaluation after epoch
                    self.validate(-1, model, sess, self.X_val, self.sent_length_val, self.word_length_val,
                                  self.y_val, self.batch_size)
                    self.HAN_embedding, self.prediction_result = [], []
                    for epoch in range(self.epochs):
                        epoch += 1

                        batches = batch_iter(list(zip(self.X_train, self.sent_length_train, self.word_length_train, self.y_train)),
                                             self.batch_size)

                        # Training loop. For each batch...
                        print('\nTraining epoch {}'.format(epoch))
                        l = []
                        a = []
                        for i, batch in tqdm(enumerate(list(batches))):
                            X_batch, sent_len_batch, word_lenght_batch, y_batch = zip(
                                *batch)
                            # print('batch_hist_v', len(batch_utt_v))
                            feed_dict = {
                                model.inputs: X_batch,
                                model.sentence_lengths: sent_len_batch,
                                model.word_lengths: word_lenght_batch,
                                model.labels: y_batch,
                                model.is_training: True,
                            }

                            if epoch == self.epochs:
                                _, step, loss, accuracy, predictions, sentence_level_output = sess.run(
                                    [model.train_op, model.global_step, model.loss, model.accuracy, model.prediction,
                                     model.sentence_level_output], feed_dict)
                                l.append(loss)
                                a.append(accuracy)
                                self.prediction_result.append(predictions)
                                self.HAN_embedding.append(sentence_level_output)
                            else:
                                _, step, loss, accuracy = sess.run(
                                    [model.train_op, model.global_step, model.loss, model.accuracy], feed_dict)
                                l.append(loss)
                                a.append(accuracy)

                        print("\t \tEpoch {}:, loss {:g}, Accuracy {:g}".format(epoch, np.average(l), np.average(a)))

                        if epoch == self.epochs:
                            self.prediction_result = np.concatenate(self.prediction_result, axis=0)
                            self.HAN_embedding = np.concatenate(self.HAN_embedding, axis=0)
                            acc = np.average(a)
                            print('acc:', acc)

                        # Evaluation after epoch
                        accuracy, report = self.validate(epoch, model, sess, self.X_val, self.sent_length_val,
                                                         self.word_length_val, self.y_val, self.batch_size)

                        if accuracy > best_acc:
                            best_epoch = epoch
                            best_acc = accuracy
                            best_report = report

                    print("\n\nBest epoch: {}\nBest test accuracy: {}".format(best_epoch, best_acc))
                    # print("\n\nBest epoch: {}\nBest test report: \n{}".format(best_epoch, best_report))
                    return self.HAN_embedding, self.prediction_result
