import gensim
from utils import *
from nltk.tokenize import word_tokenize

class Context_Embeddings(object):
    """
    Learns embeddings for NPs and relation phrases
    """

    def __init__(self, params, side_info, logger, clean_ent_list, E_init):
        self.p = params
        self.logger = logger
        self.side_info = side_info
        self.clean_ent_list = clean_ent_list
        self.E_init = E_init

    def get_naive_context_embed(self):
        self.ent2embed = []
        sentence_num, all_num, oov_num, oov_rate = 0, 0, 0, 0
        entity_oov_num, entity_oov_rate = 0, 0
        print('generate text embeddings')
        print('use pre-trained vectors:', self.p.embed_loc)
        print('use_first_sentence:', self.p.use_first_sentence)
        model = gensim.models.KeyedVectors.load_word2vec_format(self.p.embed_loc, binary=False)
        all_sentence_num = len(self.side_info.sentence_List)
        print('all sentence num:', all_sentence_num)
        unk = str('UNK')
        if self.p.use_attention:
            print('use attention')
            for i in range(len(self.clean_ent_list)):
                ent_id = self.side_info.ent2id[self.clean_ent_list[i]]
                sentence_id_list = self.side_info.ent_id2sentence_list[ent_id]
                vec_text = np.zeros(self.p.embed_dims, np.float32)
                ent_embed = self.E_init[i]
                if self.p.use_first_sentence:
                    sentence_id_list = [sentence_id_list[0]]

                for sentence_id in sentence_id_list:
                    sentence = self.side_info.sentence_List[sentence_id]
                    word_list = word_tokenize(sentence)
                    vec_sentence = np.zeros(self.p.embed_dims, np.float32)
                    for word in word_list:
                        entity_oov_cal_num = 0
                        vec = np.zeros(self.p.embed_dims, np.float32)
                        # print('word:', type(word), word)
                        all_num += 1
                        if word in model.vocab:
                            vec += model.word_vec(word)
                        else:
                            vec += np.random.randn(self.p.embed_dims)
                            # vec += model.word_vec(unk)
                            oov_num += 1
                            entity_oov_cal_num += 1
                        sim = cos_sim(ent_embed, vec)
                        # sim = 0.99 - cos_sim(ent_embed, vec)
                        if sim == 0:
                            sim = 0.0001
                        vec_sentence += (sim * vec)
                    if len(word_list) == 0:
                        word_list_length = 1
                    else:
                        word_list_length = len(word_list)
                    vec_text += (vec_sentence / word_list_length)
                if len(sentence_id_list) == 0:
                    sentence_id_list_length = 1
                else:
                    sentence_id_list_length = len(sentence_id_list)
                self.ent2embed.append(vec_text / sentence_id_list_length)
                # self.ent2embed.append((ent_embed + vec_text / sim_sum) / 2)
                sentence_num += 1
                if entity_oov_cal_num > 0:
                    entity_oov_num += 1
            oov_rate = oov_num / all_num
            entity_oov_rate = entity_oov_num / all_sentence_num
            print('oov rate:', oov_rate, 'oov num:', oov_num, 'all num:', all_num)
            print('entity oov rate:', entity_oov_rate, 'oov num:', entity_oov_num, 'all num:', all_sentence_num)
        else:
            print('do not use attention')
            for i in range(len(self.clean_ent_list)):
                ent_id = self.side_info.ent2id[self.clean_ent_list[i]]
                sentence_id_list = self.side_info.ent_id2sentence_list[ent_id]
                if self.p.use_first_sentence:
                    sentence_id_list = [sentence_id_list[0]]

                vec_text = np.zeros(self.p.embed_dims, np.float32)
                ent_embed = self.E_init[i]
                for sentence_id in sentence_id_list:
                    sentence = self.side_info.sentence_List[sentence_id]
                    vec_sentence = np.zeros(self.p.embed_dims, np.float32)
                    word_list = word_tokenize(sentence)
                    for word in word_list:
                        entity_oov_cal_num = 0
                        vec = np.zeros(self.p.embed_dims, np.float32)
                        all_num += 1
                        if word in model.vocab:
                            vec += model.word_vec(word)
                        else:
                            vec += np.random.randn(self.p.embed_dims)
                            # vec += model.word_vec(unk)
                            oov_num += 1
                            entity_oov_cal_num += 1
                        vec_sentence += vec
                    if len(word_list) == 0:
                        word_list_length = 1
                    else:
                        word_list_length = len(word_list)
                    vec_text += (vec_sentence / word_list_length)
                if len(sentence_id_list) == 0:
                    sentence_id_list_length = 1
                else:
                    sentence_id_list_length = len(sentence_id_list)
                self.ent2embed.append(vec_text / sentence_id_list_length)
                # self.ent2embed.append(ent_embed + (vec_text / word_list_length) / 2)
                sentence_num += 1
                if entity_oov_cal_num > 0:
                    entity_oov_num += 1
            oov_rate = oov_num / all_num
            entity_oov_rate = entity_oov_num / all_sentence_num
            print('oov rate:', oov_rate, 'oov num:', oov_num, 'all num:', all_num)
            print('entity oov rate:', entity_oov_rate, 'oov num:', entity_oov_num, 'all num:', all_sentence_num)
        return self.ent2embed