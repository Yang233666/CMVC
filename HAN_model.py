import tensorflow as tf
import tensorflow.contrib.layers as layers

from HAN_utils import task_specific_attention, BiGRU


class HAN_Model():
    def __init__(self,
                 vocab_size,
                 embedding_size,
                 classes,
                 word_cell,
                 sentence_cell,
                 word_output_size,
                 sentence_output_size,
                 pre_train_embedding,
                 dropout_keep_proba,
                 learning_rate=1e-4,
                 device='/cpu:0',
                 scope=None):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.classes = classes
        self.word_cell = word_cell
        self.word_output_size = word_output_size
        self.sentence_cell = sentence_cell
        self.sentence_output_size = sentence_output_size
        self.pre_train_embedding = pre_train_embedding
        self.dropout_keep_proba = dropout_keep_proba
        self.lr = learning_rate

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')

        # [document x sentence x word]
        self.inputs = tf.placeholder(shape=(None, None, None), dtype=tf.int32, name='inputs')

        # [document x sentence]
        self.word_lengths = tf.placeholder(shape=(None, None), dtype=tf.int32, name='word_lengths')

        # [document]
        self.sentence_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='sentence_lengths')

        # [document]
        self.labels = tf.placeholder(shape=(None, self.classes), dtype=tf.int32, name='labels')

        (self.document_size,
         self.sentence_size,
         self.word_size) = tf.unstack(tf.shape(self.inputs))

        # embeddings cannot be placed on GPU
        with tf.device('/cpu:0'):
            self._init_embedding(scope)

        self._init_body(scope)
        self._init_train_op()

    def _init_embedding(self, scope):
        with tf.variable_scope(scope):
            with tf.variable_scope("embedding") as scope:
                self.embedding_matrix = tf.get_variable(
                    name="embedding_matrix",
                    shape=[self.vocab_size, self.embedding_size],
                    # initializer=layers.xavier_initializer(),
                    initializer=tf.constant_initializer(self.pre_train_embedding),
                    trainable=True,
                    dtype=tf.float32)
                self.inputs_embedded = tf.nn.embedding_lookup(self.embedding_matrix, self.inputs)

    def _init_body(self, scope):
        with tf.variable_scope(scope):
            word_level_inputs = tf.reshape(self.inputs_embedded,
                                           [self.document_size * self.sentence_size, self.word_size,
                                            self.embedding_size])
            word_level_lengths = tf.reshape(self.word_lengths, [self.document_size * self.sentence_size])

            with tf.variable_scope('word'):
                word_encoder_output = BiGRU(self.word_cell, self.word_cell, word_level_inputs, word_level_lengths,
                                            name='word_BiRNN', dropout_keep_rate=self.dropout_keep_proba)

                with tf.variable_scope('attention') as scope:
                    word_level_output = task_specific_attention(word_encoder_output, self.word_output_size, scope=scope)

                with tf.variable_scope('dropout'):
                    word_level_output = layers.dropout(word_level_output, keep_prob=self.dropout_keep_proba,
                                                       is_training=self.is_training)

            # sentence_level
            sentence_inputs = tf.reshape(
                word_level_output, [self.document_size, self.sentence_size, self.word_output_size])

            with tf.variable_scope('sentence'):
                self.sentence_encoder_output = BiGRU(self.sentence_cell, self.sentence_cell, sentence_inputs,
                                                self.sentence_lengths, name='sentence_BiRNN',
                                                dropout_keep_rate=self.dropout_keep_proba)

                with tf.variable_scope('attention') as scope:
                    self.sentence_level_output = task_specific_attention(self.sentence_encoder_output, self.sentence_output_size,
                                                                    scope=scope)

                with tf.variable_scope('dropout'):
                    self.sentence_level_output = layers.dropout(self.sentence_level_output, keep_prob=self.dropout_keep_proba,
                                                           is_training=self.is_training)

            with tf.variable_scope('classifier'):
                self.logits = layers.fully_connected(self.sentence_level_output, self.classes, activation_fn=None)
                self.prediction = tf.argmax(self.logits, axis=-1)

    def _init_train_op(self):
        with tf.variable_scope('train'):
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=self.logits)

            self.loss = tf.reduce_mean(self.cross_entropy)

            self.accuracy = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(self.logits, -1), tf.argmax(self.labels, -1)), tf.float32))

            self.train_vars = tf.trainable_variables()
            reg_loss = []
            total_parameters = 0
            i = 1
            for train_var in self.train_vars:
                reg_loss.append(tf.nn.l2_loss(train_var))

                shape = train_var.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            # print('Trainable parameters:', total_parameters)
            self.loss = self.loss + 0.001 * tf.reduce_mean(reg_loss)

            opt = tf.train.AdamOptimizer(self.lr, beta1=0.9, beta2=0.999)

            self.train_op = opt.minimize(self.loss, name='train_op', global_step=self.global_step)
            # breakpoint()
