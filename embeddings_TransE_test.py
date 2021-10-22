import gensim
from helper import *
from utils import cos_sim
from cluster_f1_test import Find_Best_Result, HAC_getClusters, embed2f1, cluster_test
# from train_embedding_model import Train_Embedding_Model
from train_embedding_model_self_training import Train_Embedding_Model

class Embeddings(object):
    """
    Learns embeddings for NPs and relation phrases
    """

    def __init__(self, params, side_info, logger, true_ent2clust, true_clust2ent):
        self.p = params
        self.logger = logger

        self.side_info = side_info
        self.ent2embed = {}  # Stores final embeddings learned for noun phrases
        self.rel2embed = {}  # Stores final embeddings learned for relation phrases
        self.true_ent2clust, self.true_clust2ent = true_ent2clust, true_clust2ent

    def fit(self):
        clean_ent_list, clean_rel_list = [], []
        for ent in self.side_info.ent_list: clean_ent_list.append(ent.split('|')[0])
        for rel in self.side_info.rel_list: clean_rel_list.append(rel.split('|')[0])

        print('clean_ent_list:', type(clean_ent_list), len(clean_ent_list))  # 19915
        print('clean_rel_list:', type(clean_rel_list), len(clean_rel_list))  # 18250

        ''' Intialize embeddings '''
        if self.p.embed_init == 'crawl':
            fname1, fname2 = '../file/' + self.p.dataset + '/1E_init', '../file/' + self.p.dataset + '/1R_init'
            if not checkFile(fname1) or not checkFile(fname2):
                print('generate pre-trained embeddings')

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

        print('Test TransE Self-training ! ------------------------------------------------------------------------')
        if self.p.use_Embedding_model:
            print('generate TransE embeddings')
            print('self.training_time', 'use pre-trained crawl embeddings ... ')

            TEM = Train_Embedding_Model(self.p, self.side_info, self.logger, self.E_init, self.R_init)
            self.entity_embedding, self.relation_embedding = TEM.train()

            for id in self.side_info.id2ent.keys(): self.ent2embed[id] = self.entity_embedding[id]
            for id in self.side_info.id2rel.keys(): self.rel2embed[id] = self.relation_embedding[id]

            print('generate best clusters_list')
            cluster_threshold_max, cluster_threshold_min = 70, 10
            FBR = Find_Best_Result(self.p, self.side_info, self.logger, self.true_ent2clust, self.true_clust2ent,
                                   clean_ent_list, self.ent2embed, cluster_threshold_max, cluster_threshold_min)
            best_np_view_clusters_list, best_el_crawl_clusters_list, el_prior_cluster_list = FBR.go()
            print('best_relation_view_clusters_list')
            cluster_test(self.p, self.side_info, best_np_view_clusters_list, self.true_ent2clust, self.true_clust2ent,
                         print_or_not=True)
            exit()