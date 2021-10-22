import gc
import random
from sklearn import preprocessing
from utils import *
from base.optimizers import generate_optimizer
#tf.reset_default_graph()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    s1 = np.exp(x) - np.exp(-x)
    s2 = np.exp(x) + np.exp(-x)
    s = s1 / s2
    return s


class AutoEncoderModel:
    def __init__(self, word_vec_list, args, input_dimension=1500, hidden_dimensions=None):
        self.session = load_session()
        self.args = args
        self.weights, self.biases = {}, {}
        self.input_dimension = input_dimension
        if hidden_dimensions is None:
            hidden_dimensions = [1024, 512, self.args.dim]
            #hidden_dimensions = [512, 256, self.args.dim]
        self.hidden_dimensions = hidden_dimensions
        self.layer_num = len(self.hidden_dimensions)
        self.encoder_output = None
        self.decoder_output = None
        self.decoder_op = None

        self.word_vec_list = np.reshape(word_vec_list, [len(word_vec_list), input_dimension])
        if self.args.encoder_normalize:
            self.word_vec_list = preprocessing.normalize(self.word_vec_list)

        self._init_graph()
        self._loss_optimizer()
        tf.global_variables_initializer().run(session=self.session)

    def _init_graph(self):
        self.hidden_dimensions.insert(0, self.input_dimension)
        hds = self.hidden_dimensions
        for i in range(self.layer_num):
            self.weights['encoder_h' + str(i)] = tf.get_variable('encoder_h' + str(i),
                                                                 initializer=tf.random_normal_initializer,
                                                                 shape=[hds[i], hds[i + 1]], dtype=tf.float32)
            self.biases['encoder_b' + str(i)] = tf.get_variable('encoder_b' + str(i),
                                                                initializer=tf.random_normal_initializer,
                                                                shape=[hds[i + 1]], dtype=tf.float32)
        for i in range(self.layer_num):
            i_decoder = self.layer_num - i
            self.weights['decoder_h' + str(i)] = tf.get_variable('decoder_h' + str(i),
                                                                 initializer=tf.random_normal_initializer,
                                                                 shape=[hds[i_decoder], hds[i_decoder - 1]],
                                                                 dtype=tf.float32)

            self.biases['decoder_b' + str(i)] = tf.get_variable('decoder_b' + str(i),
                                                                initializer=tf.random_normal_initializer,
                                                                shape=[hds[i_decoder - 1]], dtype=tf.float32)
            self.batch = tf.placeholder(tf.float32, shape=[None, self.input_dimension])

    def _loss_optimizer(self):
        encoder_output = self.encoder(self.batch)
        if self.args.encoder_normalize:
            encoder_output = tf.nn.l2_normalize(encoder_output, dim=1)
        decoder_output = self.decoder(encoder_output)
        self.loss = tf.reduce_mean(tf.pow(decoder_output - self.batch, 2))# 求(decoder_output - self.batch)^2的平均值
        self.optimizer = generate_optimizer(self.loss, self.args.learning_rate, opt=self.args.optimizer)

    def encoder(self, input_data):
        input_layer = input_data
        for i in range(self.layer_num):
            input_layer = tf.add(tf.matmul(input_layer, self.weights['encoder_h' + str(i)]), self.biases['encoder_b' + str(i)])
            if self.args.encoder_active == 'sigmoid':
                input_layer = tf.nn.sigmoid(input_layer)
            elif self.args.encoder_active == 'tanh':
                input_layer = tf.nn.tanh(input_layer)
        encoder_output = input_layer
        return encoder_output

    def decoder(self, input_data):
        input_layer = input_data
        for i in range(self.layer_num):
            input_layer = tf.add(tf.matmul(input_layer, self.weights['decoder_h' + str(i)]), self.biases['decoder_b' + str(i)])
            if self.args.encoder_active == 'sigmoid':
                input_layer = tf.nn.sigmoid(input_layer)
            elif self.args.encoder_active == 'tanh':
                input_layer = tf.nn.tanh(input_layer)
        decoder_output = input_layer
        return decoder_output

    def train_one_epoch(self, epoch):
        start_time = time.time()

        batches = list()
        batch_size = self.args.batch_size
        num_batch = len(self.word_vec_list) // batch_size + 1
        for i in range(num_batch):
            if i == num_batch - 1:
                batches.append(self.word_vec_list[i * batch_size:])
            else:
                batches.append(self.word_vec_list[i * batch_size:(i + 1) * batch_size])

        self.loss_sum = 0.0
        for batch_i in range(num_batch):
            loss_train, _ = self.session.run([self.loss, self.optimizer], feed_dict={self.batch: batches[batch_i]})
            self.loss_sum += loss_train
        self.loss_sum += self.args.batch_size
        end = time.time()
        print('epoch {} of literal encoder, loss: {:.4f}, time: {:.4f}s'.format(epoch, self.loss_sum, end - start_time))
        return

    def encoder_multi_batches(self, input_data):
        print('encode literal embeddings...', len(input_data))
        batches = list()
        results = np.zeros((len(input_data), self.args.dim))
        batch_size = self.args.batch_size
        num_batch = len(input_data) // batch_size + 1
        for i in range(num_batch):
            if i == num_batch - 1:
                batches.append(input_data[i * batch_size:])
            else:
                batches.append(input_data[i * batch_size:(i + 1) * batch_size])

        for batch_i in range(num_batch):
            input_layer = np.reshape(batches[batch_i], [len(batches[batch_i]), self.input_dimension])
            for i in range(self.layer_num):
                weight_i = self.weights['encoder_h' + str(i)].eval(session=self.session)
                bias_i = self.biases['encoder_b' + str(i)].eval(session=self.session)
                input_layer = np.matmul(input_layer, weight_i) + bias_i
                if self.args.encoder_active == 'sigmoid':
                    input_layer = sigmoid(input_layer)
                elif self.args.encoder_active == 'tanh':
                    input_layer = tanh(input_layer)
            literal_vectors = input_layer
            if batch_i == num_batch - 1:
                results[batch_i * batch_size:] = np.array(literal_vectors)
            else:
                results[batch_i * batch_size:(batch_i + 1) * batch_size] = np.array(literal_vectors)
            del literal_vectors
            gc.collect()
        print("encoded literal embeddings", results.shape)
        return results

class LiteralEncoder:

    def __init__(self, args, word2vec, literal_list, literal2id, id2literal, tokens_max_len=5, word2vec_dimension=300):
        self.args = args
        self.literal_list = literal_list
        self.literal2id = literal2id
        self.id2literal = id2literal
        self.word2vec = word2vec

        def generate_word2vec_by_character_embedding(word_list, vector_dimension=300):  # 给定单词列表利用word2vec生成300维的char_embed
            character_vectors = {}
            alphabet = ''
            ch_num = {}
            for word in word_list:  # 对于单词列表中的每一个单词
                for ch in word:  # 对于单词中的每一个字符
                    n = 1
                    if ch in ch_num:
                        n += ch_num[ch]
                    ch_num[ch] = n  # 单词中的字符数
            ch_num = sorted(ch_num.items(), key=lambda x: x[1], reverse=True)  # 对ch_num中的元素按照第二个元素降序排列
            ch_sum = sum([n for (_, n) in ch_num])
            for i in range(len(ch_num)):
                if ch_num[i][1] / ch_sum >= 0.0001:
                    alphabet += ch_num[i][0]
            print(alphabet)
            print('len(alphabet):', len(alphabet), '\n')
            char_sequences = [list(word) for word in word_list]  # 提取输入的单词列表中的单词
            # print(char_sequences)
            print('char_sequences:', len(char_sequences), '\n')
            model = Word2Vec(char_sequences, size=vector_dimension, window=5, min_count=1)
            # model.save('char_embeddings.vec')
            for ch in alphabet:
                assert ch in model
                character_vectors[ch] = model[ch]  # 将word_vec生成的向量保存到char_embed
            word2vec = {}
            for word in word_list:
                vec = np.zeros(vector_dimension, dtype=np.float32)  # 向量初始化 vec为类型为np的32位浮点数组成的300维0数组
                for ch in word:
                    if ch in alphabet:
                        vec += character_vectors[ch]
                if len(word) != 0:
                    word2vec[word] = vec / len(word)
            return word2vec

        def generate_unlisted_word2vec(word2vec, literal_list):  # 将不在list里的word以word2vec嵌入并更新入word2vec
            unlisted_words = []
            for literal in literal_list:
                words = literal.split(' ')
                if str(words) in word2vec:
                    continue
                else:
                    for word in words:
                        if word not in word2vec:
                            unlisted_words.append(word)
            word2vec_char = generate_word2vec_by_character_embedding(unlisted_words)
            word2vec.update(word2vec_char)
            return word2vec

        print('word2vec', len(self.word2vec))
        self.word2vec = generate_unlisted_word2vec(self.word2vec, self.literal_list)  # 使用char
        print('word2vec', len(self.word2vec))
        self.tokens_max_len = tokens_max_len
        self.word2vec_dimension = word2vec_dimension

        literal_vector_list = []
        for id in range(len(self.id2literal)):
            literal = self.id2literal[id]
            vectors = np.zeros((self.tokens_max_len, self.word2vec_dimension), dtype=np.float32)
            words = literal.split(' ')
            for i in range(min(self.tokens_max_len, len(words))):
                vectors[i] = self.word2vec[words[i]]
            literal_vector_list.append(vectors)
        assert len(self.id2literal) == len(literal_vector_list)
        encoder_model = AutoEncoderModel(literal_vector_list, self.args)
        for i in range(self.args.encoder_epoch):
            encoder_model.train_one_epoch(i + 1)
        #literal_vector = np.array(literal_vector_list)
        #new_literal_vector = np.zeros(shape=(len(literal_vector_list), 300))
        #for i in range(len(literal_vector_list)):
            #if all(literal_vector[i][4] == 0):
                #if all(literal_vector[i][3] == 0):
                    #if all(literal_vector[i][2] == 0):
                        #if all(literal_vector[i][1] == 0):
                            #new_literal_vector[i] = literal_vector[i][0]
                        #else:
                            #new_literal_vector[i] = (literal_vector[i][0] + literal_vector[i][1]) / 2
                    #else:
                        #new_literal_vector[i] = (literal_vector[i][0] + literal_vector[i][1] + literal_vector[i][2]) / 3
                #else:
                    #new_literal_vector[i] = (literal_vector[i][0] + literal_vector[i][1] + literal_vector[i][2] +
                                             #literal_vector[i][3]) / 4
            #else:
                #new_literal_vector[i] = (literal_vector[i][0]+literal_vector[i][1]+literal_vector[i][2]+literal_vector[i][3]+literal_vector[i][4])/5
        self.encoded_literal_vector = encoder_model.encoder_multi_batches(literal_vector_list)
        print('self.encoded_literal_vector', type(self.encoded_literal_vector), self.encoded_literal_vector.shape)
        #self.encoded_literal_vector = new_literal_vector
        print('生成的向量:', type(self.encoded_literal_vector), self.encoded_literal_vector.shape)
