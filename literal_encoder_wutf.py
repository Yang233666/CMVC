import gc
import random
from sklearn import preprocessing
from utils import *
# from base.optimizers import generate_optimizer


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    s1 = np.exp(x) - np.exp(-x)
    s2 = np.exp(x) + np.exp(-x)
    s = s1 / s2
    return s


class LiteralEncoder:

    def __init__(self, args, word2vec, literal_list, literal2id, id2literal, tokens_max_len=5, word2vec_dimension=300):
        self.args = args
        self.literal_list = literal_list
        self.literal2id = literal2id
        self.id2literal = id2literal
        self.word2vec = word2vec

        def generate_unlisted_word2vec(word2vec, literal_list):  # 将不在list里的word以word2vec嵌入并更新入word2vec
            unlisted_words, listed_words = [], []
            listed_literal, literal_id_list = [], [] # 至少有一个word在word2vec的NP
            listed_literal2, literal_id_list2 = [], [] # 所有word都在word2vec的NP
            unlisted_literal, unliteral_id_list = [], []  # 没有一个word在word2vec的NP
            #print('word2vec', len(word2vec))
            for literal in literal_list:
                head, sep, tail = literal.partition('|')
                num_literal = 0
                num_literal2 = 0
                words = head.split(' ')
                if str(words) in word2vec:
                    listed_words.append(words)
                    listed_literal.append(head)
                    literal_id_list.append(literal)
                else:
                    for word in words:
                        num_literal2 += 1
                        if word not in word2vec:
                            if word not in unlisted_words:
                                unlisted_words.append(word)
                        if word in word2vec:
                            num_literal += 1
                            if word not in listed_words:
                                listed_words.append(word)
                    if num_literal > 0:
                        listed_literal.append(head)
                        literal_id_list.append(literal)
                    if num_literal == num_literal2:
                        listed_literal2.append(head)
                        literal_id_list2.append(literal)
                    if num_literal == 0:
                        unlisted_literal.append(head)
                        unliteral_id_list.append(literal)
            assert len(listed_literal) == len(literal_id_list)
            assert len(listed_literal2) == len(literal_id_list2)
            assert len(unlisted_literal) == len(unliteral_id_list)
            return listed_words, listed_literal, literal_id_list, unlisted_literal, unliteral_id_list

        listed_words, self.listed_literal_list, self.literal_id_list, self.unlisted_literal, \
        self.unliteral_id_list = generate_unlisted_word2vec(word2vec, self.literal_list)  # 不再使用char
        self.tokens_max_len = tokens_max_len
        self.word2vec_dimension = word2vec_dimension

        f = open('unlisted_literal.csv', 'w')
        for key in self.unlisted_literal:
            f.write(str(key))
            f.write('\n')
        f.close()


        literal_vector_list = []
        UNK_vec = self.word2vec['UNK']
        zero_vec = np.zeros((self.tokens_max_len, self.word2vec_dimension), dtype=np.float32)
        for id in self.id2literal.keys():
            #print('self.listed_literal_list:', self.listed_literal_list)
            #print('self.listed_literal_list:', len(self.listed_literal_list))
            literal = self.id2literal[id]
            #print('id:', id)
            #print('literal:', literal)
            if literal in self.listed_literal_list:
                # for literal in self.literal_list:
                vectors = np.zeros((self.tokens_max_len, self.word2vec_dimension), dtype=np.float32)
                words = literal.split(' ')
                for i in range(min(self.tokens_max_len, len(words))):
                    if words[i] in listed_words:
                        vectors[i] = self.word2vec[words[i]]
                    else:  # 如果单词不在word2vec里，就用word2vec里的UNK向量代替
                        vectors[i] = UNK_vec
                literal_vector_list.append(vectors)
            else:
                vectors = zero_vec
                literal_vector_list.append(vectors)
        assert len(self.id2literal) == len(literal_vector_list)
        #encoder_model = AutoEncoderModel(literal_vector_list, self.args)
        #for i in range(self.args.encoder_epoch):
            #encoder_model.train_one_epoch(i + 1)
        literal_vector = np.array(literal_vector_list)
        new_literal_vector = np.zeros(shape=(len(literal_vector_list), 300))
        for i in range(len(literal_vector_list)):
            if all(literal_vector[i][4] == 0):
                if all(literal_vector[i][3] == 0):
                    if all(literal_vector[i][2] == 0):
                        if all(literal_vector[i][1] == 0):
                            new_literal_vector[i] = literal_vector[i][0]
                        else:
                            new_literal_vector[i] = (literal_vector[i][0] + literal_vector[i][1]) / 2
                    else:
                        new_literal_vector[i] = (literal_vector[i][0] + literal_vector[i][1] + literal_vector[i][2]) / 3
                else:
                    new_literal_vector[i] = (literal_vector[i][0] + literal_vector[i][1] + literal_vector[i][2] +
                                             literal_vector[i][3]) / 4
            else:
                new_literal_vector[i] = (literal_vector[i][0]+literal_vector[i][1]+literal_vector[i][2]+literal_vector[i][3]+literal_vector[i][4])/5
        #self.encoded_literal_vector = encoder_model.encoder_multi_batches(literal_vector_list)
        #print('self.encoded_literal_vector', type(self.encoded_literal_vector), self.encoded_literal_vector.shape)
        self.encoded_literal_vector = new_literal_vector
        print('生成的向量:', type(self.encoded_literal_vector), self.encoded_literal_vector.shape)
