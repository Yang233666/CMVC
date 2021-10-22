"""
Keras implementation for Deep Embedded Clustering (DEC) algorithm:

        Junyuan Xie, Ross Girshick, and Ali Farhadi. Unsupervised deep embedding for clustering analysis. ICML 2016.

Usage:
    use `python DEC.py -h` for help.

Author:
    Xifeng Guo. 2017.1.30
"""

from time import time
import numpy as np
from numpy import asarray
from numpy import zeros
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.backend as K
# from keras.engine.topology import Layer, InputSpec
from tensorflow.keras.layers import Dense, Input, Layer, InputSpec, Lambda, Embedding, Flatten, Multiply
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD, Adam, Adagrad, Adadelta, RMSprop
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.initializers import VarianceScaling, Identity, Constant
from tensorflow.keras.utils import multi_gpu_model

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from sklearn.cluster import _k_means_fast as _k_means
from sklearn.cluster._k_means import (
    _check_sample_weight,
    _init_centroids,
    _labels_inertia,
    _tolerance,
    _validate_center_shape,
)
from cluster_f1_test import HAC_getClusters, cluster_test, embed2f1
from utils import *

# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#
# import tensorflow as tf
# print(tf.__version__)
# print('tf.test.is_gpu_available():', tf.test.is_gpu_available())

def autoencoder(dims, act='relu', init='glorot_uniform'):
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        (ae_model, encoder_model), Model of autoencoder and model of encoder
    """
    n_stacks = len(dims) - 1
    # input
    x = Input(shape=(dims[0],), name='input')
    h = x

    # hidden layer
    h = Dense(units=dims[-1], kernel_initializer=Identity(gain=1.0), name='encoder')(h)  # hidden layer, features are extracted from here

    return Model(inputs=x, outputs=h, name='encoder')

def dense_layer(dims, init='glorot_uniform'):
    # input
    # x = Input(shape=(dims[0], dims[1]), name='input')
    x = Input(shape=(dims[0], ), name='input')
    h = x
    activation = 'linear'
    # activation = 'softmax'
    # activation = 'relu'
    # activation = 'tanh'
    # activation = 'sigmoid'
    # activation = 'hard_sigmoid'
    # activation = 'elu'
    # activation = 'selu'
    # activation = 'exponential'

    # h = Dense(dims[-1], activation='linear', kernel_initializer=init, name='weight', weights=[embedding_matrix])(h)
    h = Dense(units=dims[-1], activation=activation, kernel_initializer=init, name='weight-1')(h)
    # h = Dense(dims[-1], activation=activation, kernel_initializer=init, name='weight-2')(h)

    # h = Dense(dims[-1], activation='linear', kernel_initializer=Identity(gain=1.0), name='weight')(h)
    # h = Dense(dims[-1], activation='linear', kernel_initializer=Identity(gain=1.0), name='weight')(h)
    # hidden layer
    output = h

    return Model(inputs=x, outputs=output, name='dense')


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        # self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_unitform',
        #                                 name='clusters')
        # self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), trainable=True, name='clusters')
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform',
                                        name='clusters', trainable=True)
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        # distances0 = K.expand_dims(inputs, axis=1) - self.clusters
        # c = K.sum(K.square(distances0), axis=2)
        # c = K.maximum(c, K.epsilon())
        # q = 1.0 / (1.0 + (c / self.alpha))

        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        print('input_shape:', type(input_shape), input_shape)
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def look_up_word_embedding(args, word_embedding):
    word_index = args
    print('word_index:', type(word_index), word_index.shape, word_index)
    print('word_embedding:', type(word_embedding), word_embedding.shape, word_embedding)
    np_embedding = K.dot(word_index, word_embedding)
    print('np_embedding:', np_embedding.shape, np_embedding)
    return np_embedding


class DEC(object):
    def __init__(self, params, side_info=None, word_embed=None, np_view_index=None, np_view_embed=None,
                 context_view_index=None, context_view_embed=None, dims=None, n_clusters=10,
                 alpha=1.0, true_ent2clust=None, true_clust2ent=None,
                 cluster_threshold=0.33, el_prior=None, true_answer=None):
        super(DEC, self).__init__()

        self.p = params
        self.side_info = side_info

        self.word_embed = word_embed

        self.np_view_index = np_view_index
        self.np_view_embed = np_view_embed

        self.context_view_index = context_view_index
        self.context_view_embed = context_view_embed

        self.input_x_list = [self.context_view_index, self.np_view_index]  # [(23735, 22737), (23735, 22737)]
        self.input_embed_list = [self.context_view_embed, self.np_view_embed]  # [(23735, 300), (23735, 300)]

        self.dims = dims
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.true_ent2clust, self.true_clust2ent = true_ent2clust, true_clust2ent

        self.cluster_threshold = cluster_threshold
        self.sub_entity_linking_prior = el_prior
        self.true_answer = true_answer

        # if self.p.cluster_center_init == 'random':
        #     self.np_embed = np.random.rand(self.np_embed.shape[0], self.np_embed.shape[1])
        input_word_index = Input(shape=(int(self.np_view_index.shape[-1]),), name='input_word_index')
        word_embedding = Dense(units=300, kernel_initializer=Constant(self.word_embed), trainable=False,
                               name='word_embedding')(input_word_index)
        np_view_word_embedding = ClusteringLayer(self.n_clusters, name='np_view_clustering')(word_embedding)
        self.np_view_model = Model(inputs=input_word_index, outputs=np_view_word_embedding)
        # input_context_view_word_index = Input(shape=(22737,), name='context_view_word_index')
        # word_embedding = Dense(units=300, kernel_initializer=Constant(self.word_embed), trainable=True,
        #                        name='word_embedding')(input_context_view_word_index)

        # -------------------------------
        # if self.p.cluster_method == 'hac':
        #     context_view_word_embedding = ClusteringLayer(self.n_clusters,
        #                                                   name='context_view_clustering')(word_embedding)
        #     self.context_view_model = Model(inputs=input_word_index, outputs=context_view_word_embedding)
        #     self.model_list = [self.context_view_model, self.np_view_model]
        # else:
        #     self.model_list = [self.np_view_model]
        print('DEC model init is ok ')

    # def load_weights(self, weights):  # load weights of DEC model
    #     self.np_model.load_weights(weights)
    #     self.context_model.load_weights(weights)

    # def predict(self, x):  # predict cluster labels using the output of clustering layer
    #     q = self.model.predict(x, verbose=0)
    #     print('dec predict')
    #     return q.argmax(1)

    @staticmethod
    def target_distribution(q):
        # original
        weight = q ** 2 / q.sum(0)
        td_out = (weight.T / weight.sum(1)).T
        return td_out

    def compile(self, optimizer='sgd', loss='kld'):
        self.np_view_model.compile(optimizer=optimizer, loss=loss)
        # if self.p.cluster_method == 'hac':
        #     self.context_view_model.compile(optimizer=optimizer, loss=loss)

    def get_centers(self, x, sample_weight, labels, n_clusters, distances):
        """M step of the K-means EM algorithm

        Computation of cluster centers / means.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)

        sample_weight : array-like, shape (n_samples,)
            The weights for each observation in X.

        labels : array of integers, shape (n_samples)
            Current label assignment

        n_clusters : int
            Number of desired clusters

        distances : array-like, shape (n_samples)
            Distance to closest cluster for each sample.

        Returns
        -------
        centers : array, shape (n_clusters, n_features)
            The resulting centers
        """
        n_samples = int(x.shape[0])  # 23735
        n_features = int(x.shape[1])  # 300

        dtype = np.float32
        # num = len(np.unique(labels))
        centers = np.zeros((n_clusters, n_features), dtype=dtype)  # <class 'numpy.ndarray'> (13053, 300)  fenzi
        weight_in_cluster = np.zeros((n_clusters,), dtype=dtype)  # <class 'numpy.ndarray'> (13053,)  fenmu

        # sample_weight: <class 'numpy.ndarray'> (23735,) [1. 1. 1. ... 1. 1. 1.]
        # labels <class 'numpy.ndarray'> (23735,) [5089 3052 4057 ... 8761 2708  332]

        for i in range(n_samples):
            c = labels[i]
            weight_in_cluster[c] += sample_weight[i]
        empty_clusters = np.where(weight_in_cluster == 0)[0]
        # weight_in_cluster: <class 'numpy.ndarray'> (9000,) [2. 4. 4. ... 3. 1. 1.]
        # maybe also relocate small clusters?

        if len(empty_clusters):
            # find points to reassign empty clusters to
            far_from_centers = distances.argsort()[::-1]
            # far_from_centers: <class 'numpy.ndarray'> (23735,) [ 7254 10213 15661 ... 15342 13377 19714]

            for i, cluster_id in enumerate(empty_clusters):
                # XXX two relocated clusters could be close to each other
                far_index = far_from_centers[i]
                centers[cluster_id] = x[far_index] * sample_weight[far_index]
                weight_in_cluster[cluster_id] = sample_weight[far_index]

        for i in range(n_samples):
            for j in range(n_features):
                centers[labels[i], j] += x[i, j] * sample_weight[i]

        centers /= weight_in_cluster[:, np.newaxis]
        # weight_in_cluster[:, np.newaxis]: <class 'numpy.ndarray'> (9000, 1)
        return centers

    def cluster2distribution(self, clusters):
        clusters_list = list(clusters)
        clusters_matrix = np.zeros(shape=(len(clusters_list), len(clusters_list)))
        for index in range(len(clusters_list)):
            value = clusters_list[index]
            positions_list = list(filter(lambda x: clusters_list[x] == value, list(range(len(clusters_list)))))
            for position in positions_list:
                clusters_matrix[index][position] = value
        return clusters_matrix

    def fit(self, input_list=None, train_epochs=2e4, batch_size=256, tol=1e-4,
            update_interval=140, save_dir='./results/temp'):

        # save_interval = int(x.shape[0] / batch_size) * 5  # 5 epochs
        save_interval = 100000
        print('Save interval', save_interval)

        all_f1_list = []
        print('batch_size:', batch_size)

        # weight = np.dot(self.np_view_index, self.word_embed)

        # Step 1: initialize cluster centers using k-means or HAC
        print('Initializing cluster centers with crawl embedding....')

        if self.p.cluster_method == 'kmeans':
            print('cluster_method is k-means ...')
            print('k-means n_clusters:', self.n_clusters)
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, init='k-means++', n_jobs=10)
            np_view_clusters = kmeans.fit_predict(self.np_view_embed)
            np_view_clusters_center = kmeans.cluster_centers_
            # kmeans = KMeans(n_clusters=self.context_view_n_clusters, n_init=10, init='k-means++', n_jobs=10)
            context_view_clusters = kmeans.fit_predict(self.context_view_embed)
            context_view_clusters_center = kmeans.cluster_centers_
            # kmeans.cluster_centers_   ---> np.ndarry[n_cluster, n_features] is the ave-embedding of every cluster
        else:
            print('cluster_method is HAC ...')
            print('self.cluster_threshold:', self.cluster_threshold)
            np_view_clusters, np_view_clusters_center = HAC_getClusters(self.p, self.np_view_embed, self.cluster_threshold)
            context_view_clusters, context_view_clusters_center = HAC_getClusters(self.p, self.context_view_embed,
                                                                                  self.cluster_threshold)
        ave_prec, ave_recall, ave_f1, macro_prec, micro_prec, pair_prec, macro_recall, micro_recall, \
        pair_recall, macro_f1, micro_f1, pair_f1, clusters, singletons = cluster_test(self.p, self.side_info,
                                                                                      np_view_clusters,
                                                                                      self.true_ent2clust,
                                                                                      self.true_clust2ent)
        all_f1_list.append(ave_f1)
        print('np_view_ave_f1:', ave_f1)
        ave_prec, ave_recall, ave_f1, macro_prec, micro_prec, pair_prec, macro_recall, micro_recall, \
        pair_recall, macro_f1, micro_f1, pair_f1, clusters, singletons = cluster_test(self.p, self.side_info,
                                                                                      context_view_clusters,
                                                                                      self.true_ent2clust,
                                                                                      self.true_clust2ent)
        all_f1_list.append(ave_f1)
        print('context_view_ave_f1:', ave_f1)
        np_view_clusters_center = np.random.rand(int(self.np_view_embed.shape[0]), int(self.np_view_embed.shape[1]))
        context_view_clusters_center = np.random.rand(int(self.np_view_embed.shape[0]), int(self.np_view_embed.shape[1]))
        print('np_view_clusters_center:', type(np_view_clusters_center), np_view_clusters_center.shape)
        print('context_view_clusters_center:', type(context_view_clusters_center), context_view_clusters_center.shape)

        # Step 2: deep clustering
        # logging file
        import csv
        logfile = open(save_dir + '/dec_log.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'loss'])
        logwriter.writeheader()

        print('-------------------------------------------------------------------------------------------')
        print('generate fixed p ...')
        print('self.np_view_embed:', self.np_view_embed.shape, 'self.context_view_embed:', self.context_view_embed.shape)

        # np_view_init_q = self.np_view_model.predict(self.np_view_index, verbose=1)

        # np_view_init_q = np.zeros(shape=(int(self.np_view_embed.shape[0]), self.n_clusters))  # <class 'numpy.ndarray'> (23735, 13053)
        # for i in range(len(np_view_clusters)):
        #     j = np_view_clusters[i]
        #     np_view_init_q[i][j] = 1
        # print('np_view_init_q:', type(np_view_init_q), np_view_init_q.shape, np_view_init_q)
        np_view_init_q = self.cluster2distribution(np_view_clusters)
        print('np_view_init_q:', type(np_view_init_q), np_view_init_q.shape, np_view_init_q)

        if self.p.DEC_learning_target == 'q':
            print('DEC_learning_target:', self.p.DEC_learning_target)
            np_view_init_p_01 = np.eye(np_view_init_q.shape[1])[np_view_init_q.argmax(1)]
            np_view_init_y_pred = np_view_init_p_01.argmax(1)  # <class 'numpy.ndarray'> 23735
        else:
            print('DEC_learning_target:', self.p.DEC_learning_target)
            np_view_init_p = self.target_distribution(np_view_init_q)  # update the auxiliary target distribution p
            np_view_init_p_01 = np.eye(np_view_init_p.shape[1])[np_view_init_p.argmax(1)]
            np_view_init_y_pred = np_view_init_p.argmax(1)

        y_pred_last = np_view_init_y_pred
        ave_prec, ave_recall, ave_f1, macro_prec, micro_prec, pair_prec, macro_recall, micro_recall, \
        pair_recall, macro_f1, micro_f1, pair_f1, clusters, singletons = cluster_test(self.p, self.side_info,
                                                                                      list(np_view_init_y_pred),
                                                                                      self.true_ent2clust,
                                                                                      self.true_clust2ent)
        print('Ave-prec=', ave_prec, 'macro_prec=', macro_prec, 'micro_prec=', micro_prec,
              'pair_prec=', pair_prec)
        print('Ave-recall=', ave_recall, 'macro_recall=', macro_recall, 'micro_recall=', micro_recall,
              'pair_recall=', pair_recall)
        print('Ave-F1=', ave_f1, 'macro_f1=', macro_f1, 'micro_f1=', micro_f1, 'pair_f1=', pair_f1)
        print('clusters=', clusters, 'singletons=', singletons)

        print('np_view_init_p_01:', type(np_view_init_p_01), np_view_init_p_01.shape, np_view_init_p_01)

        if self.p.use_true_answer:
            p_01_true = np.zeros_like(np_view_init_p_01)
            for row_id, column_id in enumerate(self.true_answer):
                p_01_true[row_id][column_id] = 1
            print('self.true_answer:', type(self.true_answer), len(self.true_answer), self.true_answer)

            ave_prec, ave_recall, ave_f1, macro_prec, micro_prec, pair_prec, macro_recall, micro_recall, \
            pair_recall, macro_f1, micro_f1, pair_f1, clusters, singletons = cluster_test(self.p, self.side_info,
                                                                                          self.true_answer,
                                                                                          self.true_ent2clust,
                                                                                          self.true_clust2ent)
            print('Ave-prec=', ave_prec, 'macro_prec=', macro_prec, 'micro_prec=', micro_prec,
                  'pair_prec=', pair_prec)
            print('Ave-recall=', ave_recall, 'macro_recall=', macro_recall, 'micro_recall=', micro_recall,
                  'pair_recall=', pair_recall)
            print('Ave-F1=', ave_f1, 'macro_f1=', macro_f1, 'micro_f1=', micro_f1, 'pair_f1=', pair_f1)
            print('clusters=', clusters, 'singletons=', singletons)
            print('use_true_answer . . . ')
            p = p_01_true
        else:
            print('do not use_true_answer . . . ')
            p = np_view_init_p_01  # crawl (12295, 6453) is 0-1
        print('generate fixed p is ok ...')
        print('-------------------------------------------------------------------------------------------')
        print()

        num = 1
        # x = np.random.rand(x.shape[0], x.shape[1])
        # x = self.input
        distance_metric = 'euclidean'

        # ---------------------------- model's input is index ---------------------------

        self.np_view_model.get_layer(name='np_view_clustering').set_weights([np_view_clusters_center])
        # self.context_view_model.get_layer(name='context_view_clustering').set_weights([context_view_clusters_center])
        # self.context_view_model.get_layer(name='context_view_clustering').set_weights([np_view_clusters_center])
        if self.p.cluster_method == 'hac':
            # distances = np.zeros(shape=(self.context_view_embed.shape[0],),
            #                      dtype=self.context_view_embed.dtype)  # <class 'numpy.ndarray'> (23735,)
            if distance_metric == 'euclidean':
                labels, distances = pairwise_distances_argmin_min(
                    X=self.context_view_embed, Y=context_view_clusters_center, metric='euclidean',
                    metric_kwargs={'squared': True})
            else:
                labels, distances = pairwise_distances_argmin_min(
                    X=self.context_view_embed, Y=context_view_clusters_center, metric='cosine', metric_kwargs=None)

            sample_weight = _check_sample_weight(None,
                                                 self.context_view_embed)  # <class 'numpy.ndarray'> [1. 1. 1. ... 1. 1. 1.]
            new_clusters_center = self.get_centers(
                self.context_view_embed.astype(np.float),
                sample_weight.astype(np.float),
                np_view_clusters,
                self.n_clusters,
                distances.astype(np.float))
            self.context_view_model.get_layer(name='context_view_clustering').set_weights([new_clusters_center])

        loss = 0
        index = 0
        # x_length = x.shape[0]
        x_length = len(self.np_view_index)  # 23735
        index_array = np.arange(x_length)

        # tbcallback = TensorBoard(log_dir='./vis')

        print('------------------------------------------------------------------------------------------')
        print('Train begin ...')
        print('------------------------------------------------------------------------------------------')
        print()

        # self.paraller_model.fit(x=x, y=p, epochs=1, batch_size=256, verbose=1)
        # self.model.fit(x=x, y=p, epochs=num, batch_size=4, verbose=1)

        # update_interval = int(update_interval * 48)
        print('update_interval:', update_interval)
        updata_p = False
        # updata_p = True
        print('updata_p:', updata_p)

        if self.p.use_Entity_linking_dict:
            print('use_Entity_linking_dict . . . ')
            p = np.zeros_like(np_view_init_q)
            for i in range(len(np_view_init_q)):
                el_prior = self.sub_entity_linking_prior[i]
                p[i] = np_view_init_q[el_prior]
            p = np.eye(p.shape[1])[p.argmax(1)]

            y_pred = p.argmax(1)
            y_pred_last = np.copy(y_pred)
            cluster_predict_list = list(y_pred)
            ave_prec, ave_recall, ave_f1, macro_prec, micro_prec, pair_prec, macro_recall, micro_recall, \
            pair_recall, macro_f1, micro_f1, pair_f1, clusters, singletons = cluster_test(self.p,
                                                                                          self.side_info,
                                                                                          cluster_predict_list,
                                                                                          self.true_ent2clust,
                                                                                          self.true_clust2ent)
            all_f1_list.append(ave_f1)
            # print('P-Iter %d: ' % ite, ' ; loss=', loss)
            print('y_pred:', len(list(set(y_pred))), y_pred)
            print('Ave-prec=', ave_prec, 'macro_prec=', macro_prec, 'micro_prec=', micro_prec,
                  'pair_prec=', pair_prec)
            print('Ave-recall=', ave_recall, 'macro_recall=', macro_recall, 'micro_recall=', micro_recall,
                  'pair_recall=', pair_recall)
            print('Ave-F1=', ave_f1, 'macro_f1=', macro_f1, 'micro_f1=', micro_f1, 'pair_f1=', pair_f1)
            print('clusters=', clusters, 'singletons=', singletons)

        num = int(x_length / batch_size) + 1
        for ite in range(int(train_epochs)):
            for j in range(len(self.input_x_list)):
                # print('ite:', ite, 'j:', j)
                if not self.p.cluster_method == 'hac':
                    if j == 1:
                        continue
                p_old = p.copy()
                p_old = np.eye(p_old.shape[1])[p_old.argmax(1)]

                x = self.input_x_list[j]
                embed = self.input_embed_list[j]
                model = self.model_list[j]

                if ite % update_interval == 0:
                    print('-------------------------------------------------------------------------------------------')
                    q = model.predict(x, verbose=0)
                    q = np.eye(q.shape[1])[q.argmax(1)]
                    y_pred = q.argmax(1)
                    ave_prec, ave_recall, ave_f1, macro_prec, micro_prec, pair_prec, macro_recall, micro_recall, \
                    pair_recall, macro_f1, micro_f1, pair_f1, clusters, singletons = cluster_test(self.p,
                                                                                                  self.side_info,
                                                                                                  list(y_pred),
                                                                                                  self.true_ent2clust,
                                                                                                  self.true_clust2ent)
                    print('raw Q Ave-F1=', ave_f1, 'macro_f1=', macro_f1, 'micro_f1=', micro_f1, 'pair_f1=', pair_f1,
                          'clusters=', clusters, 'singletons=', singletons)

                # M step: computation of the means
                if self.p.update_center:
                    if self.p.cluster_method == 'hac':
                        if j == 0:
                            clusters_center_list = model.get_layer(name='context_view_clustering').get_weights()
                        else:
                            clusters_center_list = model.get_layer(name='np_view_clustering').get_weights()
                    else:
                        clusters_center_list = model.get_layer(name='np_view_clustering').get_weights()
                    clusters_center = clusters_center_list[0]

                    if distance_metric == 'euclidean':
                        labels, distances = pairwise_distances_argmin_min(
                            X=embed, Y=clusters_center, metric='euclidean', metric_kwargs={'squared': True})
                    else:
                        labels, distances = pairwise_distances_argmin_min(
                            X=embed, Y=clusters_center, metric='cosine', metric_kwargs=None)
                    sample_weight = _check_sample_weight(None, x)  # <class 'numpy.ndarray'> [1. 1. 1. ... 1. 1. 1.]
                    clusters_center = self.get_centers(
                        embed.astype(np.float),
                        sample_weight.astype(np.float),
                        p_old.argmax(1),
                        self.n_clusters,
                        distances.astype(np.float))

                    # l2-normalize centers (this is the main contribution here) 3
                    # print('----------', 'clusters_center:', type(clusters_center), clusters_center.shape)
                    clusters_center = normalize(clusters_center)

                    # 2=0.3654 2+3=0.5904
                    if self.p.cluster_method == 'hac':
                        if j == 0:
                            model.get_layer(name='context_view_clustering').set_weights([clusters_center])
                        else:
                            model.get_layer(name='np_view_clustering').set_weights([clusters_center])
                    else:
                        model.get_layer(name='np_view_clustering').set_weights([clusters_center])

                # if ite % update_interval == 0 or ite > 0:
                # if ite % update_interval == 0:
                #     # if j == 0:
                #     #     print('context view:')
                #     # else:
                #     #     print('np view:')
                #     # print('Test', 'updata_p:', updata_p, 'tol:', tol)
                #     # print('---------------------------------------------------------------------------------------')
                #     q = model.predict(x, verbose=0)
                #
                #     # get model.layers weight
                #     # names = [weight.name for layer in self.model.layers for weight in layer.weights]
                #     # weights = self.model.get_weights()
                #     # for name, weight in zip(names, weights):
                #     #     print(name, weight.shape, weight)
                #
                #     if self.p.use_Entity_linking_dict:
                #         p = np.zeros_like(q)
                #         for i in range(len(q)):
                #             el_prior = self.sub_entity_linking_prior[i]
                #             p[i] = q[el_prior]
                #         q = np.eye(p.shape[1])[p.argmax(1)]
                #     else:
                #         q = np.eye(q.shape[1])[q.argmax(1)]
                #
                #     if updata_p:  # use DEC's P to be learning target
                #         update_interval = int(100 * 48)
                #         p = self.target_distribution(q)  # update the auxiliary target distribution p
                #         # print('p:', type(p), p.shape, p)
                #         p = np.eye(p.shape[1])[p.argmax(1)]
                #         # print('p:', type(p), p.shape, p)
                #         y_pred = p.argmax(1)
                #         y_pred_last = np.copy(y_pred)
                #         # logdict = dict(iter=ite, loss=loss)
                #         # logwriter.writerow(logdict)
                #         cluster_predict_list = list(y_pred)
                #         ave_prec, ave_recall, ave_f1, macro_prec, micro_prec, pair_prec, macro_recall, micro_recall, \
                #         pair_recall, macro_f1, micro_f1, pair_f1, clusters, singletons = cluster_test(self.p,
                #                                                                                       self.side_info,
                #                                                                                       cluster_predict_list,
                #                                                                                       self.true_ent2clust,
                #                                                                                       self.true_clust2ent)
                #         all_f1_list.append(ave_f1)
                #         # print('P-Iter %d: ' % (ite * num / 48), ' ; loss=', loss)
                #         print('P-Iter %d: ' % ite, ' ; loss=', loss)
                #         print('y_pred:', len(list(set(y_pred))), y_pred)
                #         print('Ave-prec=', ave_prec, 'macro_prec=', macro_prec, 'micro_prec=', micro_prec,
                #               'pair_prec=', pair_prec)
                #         print('Ave-recall=', ave_recall, 'macro_recall=', macro_recall, 'micro_recall=', micro_recall,
                #               'pair_recall=', pair_recall)
                #         print('Ave-F1=', ave_f1, 'macro_f1=', macro_f1, 'micro_f1=', micro_f1, 'pair_f1=', pair_f1)
                #         print('clusters=', clusters, 'singletons=', singletons)
                #
                #     # evaluate the clustering performance
                #     y_pred = q.argmax(1)
                #     # print('y_pred:', type(y_pred), len(y_pred), y_pred)
                #     loss = np.round(loss, 5)
                #     # print('0loss:', type(loss), loss.shape, loss)
                #     # logdict = dict(iter=ite, loss=loss)
                #     # logwriter.writerow(logdict)
                #
                #     ave_prec, ave_recall, ave_f1, macro_prec, micro_prec, pair_prec, macro_recall, micro_recall, \
                #     pair_recall, macro_f1, micro_f1, pair_f1, clusters, singletons = cluster_test(self.p,
                #                                                                                   self.side_info,
                #                                                                                   list(y_pred),
                #                                                                                   self.true_ent2clust,
                #                                                                                   self.true_clust2ent)
                #     all_f1_list.append(ave_f1)
                #     # print('Q-Iter %d: ' % (ite * num / 48), ' ; loss=', loss)
                #     print('Q-Iter %d: ' % ite, ' ; loss=', loss)
                #     print('Ave-F1=', ave_f1, 'macro_f1=', macro_f1, 'micro_f1=', micro_f1, 'pair_f1=', pair_f1,
                #           'clusters=', clusters, 'singletons=', singletons)
                #
                #     # check stop criterion
                #     delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                #     # print('delta_label:', delta_label, 'tol:', tol)
                #     # print('---------------------------------------------------------------------------------------')
                #     # print()
                #     # if updata_p:
                #     #     y_pred_last = np.copy(y_pred)
                #
                #     if ite > 0 and delta_label < tol and updata_p:
                #         print('delta_label ', delta_label, '< tol ', tol)
                #         print('Reached tolerance threshold. Stopping training.')
                #         logfile.close()
                #         break

                    # if ite > 0 and delta_label < tol and not updata_p:
                    #     print('delta_label ', delta_label, '< tol ', tol)
                    #     updata_p = True
                    #     tol = tol / 60
                    #     print('updata_p:', updata_p, 'new tol:', tol)
                    #     print('==========================begin to update p =====================================')

                # train on batch
                index = 0
                for i in range(num):
                    if index == 0:
                        np.random.shuffle(index_array)
                    idx = index_array[index * batch_size: min((index + 1) * batch_size, x_length)]
                    loss = model.train_on_batch(x=x[idx], y=p_old[idx])
                    index = index + 1 if (index + 1) * batch_size <= x_length else 0
                if ite % update_interval == 0:
                    q = model.predict(x=x, batch_size=256, verbose=0)
                    p = np.eye(q.shape[1])[q.argmax(1)]
                    y_pred = p.argmax(1)
                    ave_prec, ave_recall, ave_f1, macro_prec, micro_prec, pair_prec, macro_recall, micro_recall, \
                    pair_recall, macro_f1, micro_f1, pair_f1, clusters, singletons = cluster_test(self.p,
                                                                                                  self.side_info,
                                                                                                  list(y_pred),
                                                                                                  self.true_ent2clust,
                                                                                                  self.true_clust2ent)
                    all_f1_list.append(ave_f1)
                    # print('P-Iter %d: ' % (ite * num / 48), ' ; loss=', loss)
                    print('P-Iter %d: ' % ite, 'j:', j, ' ; loss=', loss)
                    print('Ave-F1=', ave_f1, 'macro_f1=', macro_f1, 'micro_f1=', micro_f1, 'pair_f1=', pair_f1,
                          'clusters=',
                          clusters, 'singletons=', singletons)
                    print('-------------------------------------------------------------------------------------------')
                    print()

                # save intermediate model
                # if ite % save_interval == 0:
                #     print('saving model to:', save_dir + '/DEC_model_' + str(ite) + '.h5')
                #     self.model.save_weights(save_dir + '/DEC_model_' + str(ite) + '.h5')
                # ite += 1

        # save the trained model
        logfile.close()
        # print('saving model to:', save_dir + '/DEC_model_final.h5')
        # self.model.save_weights(save_dir + '/DEC_model_final.h5')
        print('-------------------------------------------------------------------------------------------')
        print('Train end ...')
        print('-------------------------------------------------------------------------------------------')
        print('all_f1_list:', type(all_f1_list), len(all_f1_list), all_f1_list)

        return y_pred, all_f1_list

class DEC_multi_view(object):
    def __init__(self, params, side_info, word_embed, np_index, np_view_embed, context_index, context_view_embed,
                 true_ent2clust, true_clust2ent, el_prior, true_answer):
        self.p = params
        self.side_info = side_info
        self.word_embed = word_embed
        self.np_index = np_index
        self.context_index = context_index
        self.np_view_embed = np_view_embed
        self.context_view_embed = context_view_embed
        self.batch_size = self.p.batch_size
        self.lr = self.p.lr
        self.embed_dim = self.p.embed_dims
        self.true_ent2clust, self.true_clust2ent = true_ent2clust, true_clust2ent
        self.entity_linking_prior = el_prior
        self.true_answer = true_answer
        # self.np_embed = np.dot(self.np_index, self.word_embed)
        # self.context_embed = np.dot(self.context_index, self.word_embed)
        self.optimizer = SGD(0.001, 0.9, nesterov=False)

        import os
        if not os.path.exists(self.p.save_dir):
            os.makedirs(self.p.save_dir)

        if self.p.metric == 'cosine':
            self.cluster_threshold = 0.33
        else:
            self.cluster_threshold = 3.44

        if self.p.cluster_method == 'hac':
            # np_view_clusters, np_view_clusters_center = HAC_getClusters(self.p, self.np_view_embed,
            #                                                             self.cluster_threshold)
            # context_view_clusters, context_view_clusters_center = HAC_getClusters(self.p, self.context_view_embed,
            #                                                                       self.cluster_threshold)
            # self.np_view_n_clusters = int(np_view_clusters_center.shape[0])
            # self.context_view_n_clusters = int(context_view_clusters_center.shape[0])
            # self.np_view_n_clusters = 13053
            # self.context_view_n_clusters = 11893
            # print('self.np_view_n_clusters:', self.np_view_n_clusters)
            # print('self.context_view_n_clusters:', self.context_view_n_clusters)
            self.n_clusters = int(self.np_view_embed.shape[0])
            print('self.n_clusters:', self.n_clusters)
        else:
            # self.np_view_n_clusters = 13053
            # self.context_view_n_clusters = 13053
            # print('self.np_view_n_clusters:', self.np_view_n_clusters)
            # print('self.context_view_n_clusters:', self.context_view_n_clusters)
            self.n_clusters = int(self.np_view_embed.shape[0])
            print('self.n_clusters:', self.n_clusters)
            # tiao can
            # cluster_threshold_max, cluster_threshold_min = 40, 20
            # context_best_threshold = dict()
            # context_best_cluster_threshold, context_best_ave_f1 = 0, 0
            # for cluster_threshold in range(cluster_threshold_max, cluster_threshold_min, -1):
            #     self.cluster_threshold = cluster_threshold / 100
            #     context_ave_f1 = embed2f1(params=self.p, embed=self.context_view,
            #                               cluster_threshold_real=self.cluster_threshold, side_info=self.side_info,
            #                               true_ent2clust=self.true_ent2clust, true_clust2ent=self.true_clust2ent)
            #     context_best_threshold.update({self.cluster_threshold: context_ave_f1})
            # for cluster_threshold in range(cluster_threshold_max, cluster_threshold_min, -1):
            #     cluster_threshold_real = cluster_threshold / 100
            #     value = context_best_threshold[cluster_threshold_real]
            #     if value > context_best_ave_f1:
            #         context_best_cluster_threshold = cluster_threshold_real
            #         context_best_ave_f1 = value
            #     else:
            #         continue
            # print('context best_threshold_dict:', context_best_threshold)
            # print('context best_cluster_threshold:', context_best_cluster_threshold, 'context best_ave_f1:', context_best_ave_f1)
            # print('context view:')
            # np best_cluster_threshold: 0.33 np best_ave_f1: 0.7741000000000001
            # if context_embed = S_init context best_cluster_threshold: 0.28 context best_ave_f1: 0.6019
            # if context_embed = (E_init + S_init) / 2 context best_cluster_threshold: 0.34 context best_ave_f1: 0.77
        self.y_pred, self.all_f1_list = self.train()

    def train(self):
        # self.dims = [self.input.shape[0], self.input.shape[1]]
        self.dims = [self.np_view_embed.shape[-1], 300]  # [300, 300]
        # print('self.dims:', self.dims)
        # prepare the DEC model
        dec = DEC(params=self.p, side_info=self.side_info, word_embed=self.word_embed,
                  np_view_index=self.np_index, np_view_embed=self.np_view_embed,
                  context_view_index=self.context_index, context_view_embed=self.context_view_embed, dims=self.dims,
                  n_clusters=self.n_clusters,
                  true_ent2clust=self.true_ent2clust, true_clust2ent=self.true_clust2ent,
                  cluster_threshold=self.cluster_threshold, el_prior=self.entity_linking_prior,
                  true_answer=self.true_answer)
        dec.np_view_model.summary()
        print()
        if self.p.cluster_method == 'hac':
            dec.context_view_model.summary()
            print()
        t0 = time.time()

        dec.compile(optimizer=self.optimizer, loss='kld')
        # dec.compile(optimizer=self.optimizer, loss='categorical_crossentropy')
        # dec.compile(optimizer=SGD(0.1, 0.9, nesterov=False), loss='kld')
        # dec.compile(optimizer=SGD(0.01, 0.9), loss='categorical_crossentropy')

        self.y_pred, self.all_f1_list = dec.fit(input_list=self.word_embed, tol=self.p.tol,
                                                train_epochs=self.p.train_epochs, batch_size=self.p.batch_size,
                                                update_interval=self.p.update_interval, save_dir=self.p.save_dir)
        time_cost = time.time() - t0
        print('clustering time: ', time_cost/3600, 'h')
        return self.y_pred, self.all_f1_list