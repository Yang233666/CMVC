import gensim
from helper import *
from utils import *


class Embeddings(object):
    """
    Learns embeddings for NPs and relation phrases
    """

    def __init__(self, params, side_info, logger):
        self.p = params
        self.logger = logger

        self.side_info = side_info
        self.ent2embed = {}  # Stores final embeddings learned for noun phrases
        self.rel2embed = {}  # Stores final embeddings learned for relation phrases
        self.ent2dict_embed = {}

    def fit(self):
        clean_ent_list, clean_rel_list, clean_sub_list, clean_obj_list = [], [], [], []
        if self.p.use_Entity_linking_dict:
            for ent in self.side_info.new_ent_list: clean_ent_list.append(ent.split('|')[0])
            for rel in self.side_info.new_rel_list: clean_rel_list.append(rel.split('|')[0])
        else:
            for ent in self.side_info.ent_list: clean_ent_list.append(ent.split('|')[0])
            for rel in self.side_info.rel_list: clean_rel_list.append(rel.split('|')[0])
            for sub in self.side_info.sub_list: clean_sub_list.append(sub.split('|')[0])
            for obj in self.side_info.obj_list: clean_obj_list.append(obj.split('|')[0])

        ''' Intialize embeddings '''
        if self.p.embed_init == 'crawl':
            fname1, fname2 = './file/1E_init', './file/1R_init'
            if not checkFile(fname1) or not checkFile(fname2):
                print('generate init word2vec dict embeddings')
                print('use pre-trained vectors:', self.p.embed_loc)
                model = gensim.models.KeyedVectors.load_word2vec_format(self.p.embed_loc, binary=False)
                self.E_init = getEmbeddings(model, clean_ent_list, self.p.embed_dims)
                self.R_init = getEmbeddings(model, clean_rel_list, self.p.embed_dims)

                pickle.dump(self.E_init, open(fname1, 'wb'))
                pickle.dump(self.R_init, open(fname2, 'wb'))

            else:
                print('load init embeddings')
                self.E_init = pickle.load(open(fname1, 'rb'))
                self.R_init = pickle.load(open(fname2, 'rb'))

        else:
            print('generate init random embeddings')
            self.E_init = np.random.rand(len(clean_ent_list), self.p.embed_dims)
            self.R_init = np.random.rand(len(clean_rel_list), self.p.embed_dims)

        if self.p.use_Entity_linking_dict:
            for id in self.side_info.id2ent.keys():
                entity_id = self.side_info.ent_old_id2new_id[id]
                entity = self.side_info.id2ent[entity_id]
                new_id = self.side_info.new_ent2id[entity]
                self.ent2embed[id] = self.E_init[new_id]
            for id in self.side_info.id2rel.keys():
                relation_id = self.side_info.rel_old_id2new_id[id]
                relation = self.side_info.id2rel[relation_id]
                new_id = self.side_info.new_rel2id[relation]
                self.rel2embed[id] = self.R_init[new_id]
        else:
            for id in self.side_info.id2ent.keys(): self.ent2embed[id] = self.E_init[id]
            for id in self.side_info.id2rel.keys(): self.rel2embed[id] = self.R_init[id]
        #for id in self.side_info.id2ent.keys(): self.ent2embed[id] = entity_embedding[id]
        #for id in self.side_info.id2rel.keys(): self.rel2embed[id] = relation_embedding[id]

        #for id in self.side_info.id2ent.keys(): self.ent2dict_embed[id] = self.E_init[id]


