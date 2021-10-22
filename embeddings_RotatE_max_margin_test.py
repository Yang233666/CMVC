import gensim
from helper import *
from utils import cos_sim
from cluster_f1_test import Find_Best_Result, HAC_getClusters, embed2f1, cluster_test
from get_context_view_embedding import Context_Embeddings
from train_embedding_model import Train_Embedding_Model
from BERT_Embeddings import BERT_Model
import collections

def get_EL_seed_pair(ent_list, ent2id, ent_old_id2new_id):
    EL_seed = []
    for i in range(len(ent_list)):
        ent1 = ent_list[i]
        old_id1 = ent2id[ent1]
        for j in range(i + 1, len(ent_list)):
            ent2 = ent_list[j]
            old_id2 = ent2id[ent2]
            new_id1, new_id2 = ent_old_id2new_id[old_id1], ent_old_id2new_id[
                old_id2]
            if new_id1 == new_id2:
                id_tuple = (i, j)
                EL_seed.append(id_tuple)
    return EL_seed

def difference_cluster2pair(cluster_list_1, cluster_list_2, EL_seed):
    # print('cluster_list_1:', type(cluster_list_1), len(cluster_list_1), cluster_list_1)
    # print('cluster_list_2:', type(cluster_list_2), len(cluster_list_2), cluster_list_2)
    new_seed_pair_list = []
    for i in range(len(cluster_list_1)):
        id_1, id_2 = cluster_list_1[i], cluster_list_2[i]
        if id_1 == id_2:continue
        else:
            index_list_1 = [i for i, x in enumerate(cluster_list_1) if x == id_1]
            index_list_2 = [i for i, x in enumerate(cluster_list_2) if x == id_2]
            if len(index_list_2) == 1:continue
            else:
                iter_list_1 = list(itertools.combinations(index_list_1, 2))
                iter_list_2 = list(itertools.combinations(index_list_2, 2))
                if len(iter_list_1) > 0:
                    for iter_pair in iter_list_1:
                        if iter_pair in iter_list_2:iter_list_2.remove(iter_pair)
                for iter in iter_list_2:
                    if iter not in EL_seed:
                        new_seed_pair_list.append(iter)
    return new_seed_pair_list

def totol_cluster2pair(cluster_list):
    # print('cluster_list:', type(cluster_list), len(cluster_list), cluster_list)
    new_seed_pair_list = []
    for i in range(len(cluster_list)):
        id = cluster_list[i]
        index_list = [i for i, x in enumerate(cluster_list) if x == id]
        if len(index_list) > 1:
            iter_list = list(itertools.combinations(index_list, 2))
            new_seed_pair_list += iter_list
    return new_seed_pair_list

def pair2triples(seed_pair_list, ent_list, ent2id, id2ent, ent2triple_id_list, trpIds, E_init, cos_sim, high_confidence=False):
    seed_trpIds, seed_sim = [], []
    for seed_pair in seed_pair_list:
        i, j = seed_pair[0], seed_pair[1]
        ent1, ent2 = ent_list[i], ent_list[j]
        if not np.dot(E_init[i], E_init[j]) == 0:sim = cos_sim(E_init[i], E_init[j])
        else:sim = 0
        if high_confidence:
            if sim > 0.9:
                Append = True
            else:
                Append = False
        else:
            Append = True
        if Append:
            for ent in [ent1, ent2]:
                triple_list = ent2triple_id_list[ent]
                for triple_id in triple_list:
                    triple = trpIds[triple_id]
                    if str(id2ent[triple[0]]) == str(ent1):
                        trp = (ent2id[str(ent2)], triple[1], triple[2])
                        seed_trpIds.append(trp)
                        seed_sim.append(sim)
                    if str(id2ent[triple[0]]) == str(ent2):
                        trp = (ent2id[str(ent1)], triple[1], triple[2])
                        seed_trpIds.append(trp)
                        seed_sim.append(sim)
                    if str(id2ent[triple[2]]) == str(ent1):
                        trp = (triple[0], triple[1], ent2id[str(ent2)])
                        seed_trpIds.append(trp)
                        seed_sim.append(sim)
                    if str(id2ent[triple[2]]) == str(ent2):
                        trp = (triple[0], triple[1], ent2id[str(ent1)])
                        seed_trpIds.append(trp)
                        seed_sim.append(sim)
    return seed_trpIds, seed_sim

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

        fname1 = '../file/' + self.p.dataset + '/EL_seed'
        if not checkFile(fname1):
            self.EL_seed = get_EL_seed_pair(self.side_info.ent_list, self.side_info.ent2id, self.side_info.ent_old_id2new_id)
            pickle.dump(self.EL_seed, open(fname1, 'wb'))
        else:
            self.EL_seed = pickle.load(open(fname1, 'rb'))
        print('self.EL_seed:', type(self.EL_seed), len(self.EL_seed))

        # if self.p.use_context:
        #     if self.p.use_BERT:
        #         self.entity_view_embed = []
        #         for ent in clean_ent_list:
        #             id = self.side_info.ent2id[ent]
        #             if id in self.side_info.isSub:
        #                 self.entity_view_embed.append(self.E_init[id])
        #
        #         if self.p.dataset == 'OPIEC':
        #             cluster_threshold = 0.49
        #         else:
        #             cluster_threshold = 0.33
        #         embed2f1(self.p, self.entity_view_embed, cluster_threshold, self.side_info, self.true_ent2clust, self.true_clust2ent,
        #                  dim_is_bert=False)
        #         clusters, clusters_center = HAC_getClusters(self.p, self.entity_view_embed, cluster_threshold)
        #         cluster_predict_list = list(clusters)
        #         BM = BERT_Model(self.p, self.side_info, self.logger, clean_ent_list, cluster_predict_list, self.true_ent2clust, self.true_clust2ent)
        #         BM.fine_tune()
        if self.p.use_context and not self.p.use_BERT:
            CE = Context_Embeddings(self.p, self.side_info, self.logger, clean_ent_list, self.E_init)
            self.ent2embed = CE.get_naive_context_embed()
            cluster_threshold_max, cluster_threshold_min = 70, 10
            FBR = Find_Best_Result(self.p, self.side_info, self.logger, self.true_ent2clust, self.true_clust2ent,
                                   clean_ent_list, self.ent2embed, cluster_threshold_max, cluster_threshold_min)
            FBR.go()
            real_time = time.strftime("%Y_%m_%d") + ' ' + time.strftime("%H:%M:%S")
            print('time:', real_time)
            exit()

        use_bert_update_seeds, use_new_embedding = self.p.use_bert_update_seeds, self.p.use_new_embedding
        print('use_bert_update_seeds:', use_bert_update_seeds, 'use_new_embedding:', use_new_embedding)
        if use_bert_update_seeds:
            if use_new_embedding:
                if self.p.use_first_sentence:
                    folder = 'BERT_first_new'
                else:
                    folder = 'BERT_all_new'
            else:
                if self.p.use_first_sentence:
                    folder = 'BERT_first'
                else:
                    folder = 'BERT_all'
        else:
            if use_new_embedding:
                folder = 'TransE_new'
            else:
                folder = 'TransE'
        print('folder:', folder)
        for i in range(10):
            self.training_time = i
            print('self.training_time:', self.training_time)

            if self.p.use_Embedding_model:
                fname1, fname2 = '../file/' + self.p.dataset + '/' + folder + '/entity_embedding_' + str(
                    self.training_time), '../file/' + self.p.dataset + '/' + folder + '/relation_embedding_' + str(self.training_time)
                if not checkFile(fname1) or not checkFile(fname2):
                    print('generate TransE embeddings', self.training_time, fname1)
                    self.generate = True
                    if self.training_time == 0:
                        self.new_seed_trpIds, self.new_seed_sim = [], []
                        entity_embedding, relation_embedding = self.E_init, self.R_init
                        print('self.training_time', self.training_time, 'use pre-trained crawl embeddings ... ')
                    else:
                        if use_new_embedding:
                            entity_embedding, relation_embedding = self.entity_embedding, self.relation_embedding
                            print('self.training_time', self.training_time, 'use trained TransE embeddings ... ')
                        else:
                            entity_embedding, relation_embedding = self.E_init, self.R_init
                            print('self.training_time', self.training_time, 'use pre-trained crawl embeddings ... ')
                    TEM = Train_Embedding_Model(self.p, self.side_info, self.logger, entity_embedding, relation_embedding,
                                                self.new_seed_trpIds, self.new_seed_sim)
                    self.entity_embedding, self.relation_embedding = TEM.train()

                    pickle.dump(self.entity_embedding, open(fname1, 'wb'))
                    pickle.dump(self.relation_embedding, open(fname2, 'wb'))
                else:
                    print('load TransE embeddings', self.training_time)
                    self.entity_embedding = pickle.load(open(fname1, 'rb'))
                    self.relation_embedding = pickle.load(open(fname2, 'rb'))
                    self.generate = False

                for id in self.side_info.id2ent.keys(): self.ent2embed[id] = self.entity_embedding[id]
                for id in self.side_info.id2rel.keys(): self.rel2embed[id] = self.relation_embedding[id]

            else:  # do not use embedding model
                for id in self.side_info.id2ent.keys(): self.ent2embed[id] = self.E_init[id]
                for id in self.side_info.id2rel.keys(): self.rel2embed[id] = self.R_init[id]

            fname1, fname2 = '../file/' + self.p.dataset + '/' + folder + '/best_np_view_clusters_list_' + str(
                self.training_time), '../file/' + self.p.dataset + '/' + folder + '/best_el_crawl_cluster_list_' + str(
                self.training_time)
            fname3 = '../file/' + self.p.dataset + '/' + folder + '/el_prior_cluster_list'
            if not checkFile(fname1) or not checkFile(fname2):
                print('generate best clusters_list', self.training_time, fname1)
                cluster_threshold_max, cluster_threshold_min = 70, 10
                FBR = Find_Best_Result(self.p, self.side_info, self.logger, self.true_ent2clust, self.true_clust2ent,
                                       clean_ent_list, self.ent2embed, cluster_threshold_max, cluster_threshold_min)
                best_np_view_clusters_list, best_el_crawl_clusters_list, el_prior_cluster_list = FBR.go()
                if not checkFile(fname3):
                    pickle.dump(el_prior_cluster_list, open(fname3, 'wb'))
                pickle.dump(best_np_view_clusters_list, open(fname1, 'wb'))
                pickle.dump(best_el_crawl_clusters_list, open(fname2, 'wb'))
            else:
                print('load best clusters_list', self.training_time)
                best_np_view_clusters_list = pickle.load(open(fname1, 'rb'))
                best_el_crawl_clusters_list = pickle.load(open(fname2, 'rb'))
                el_prior_cluster_list = pickle.load(open(fname3, 'rb'))
            print('el_prior_cluster_list')
            cluster_test(self.p, self.side_info, el_prior_cluster_list, self.true_ent2clust, self.true_clust2ent, print_or_not=True)
            # print('best_np_view_clusters_list')
            # cluster_test(self.p, self.side_info, best_np_view_clusters_list, self.true_ent2clust, self.true_clust2ent, print_or_not=True)
            # print('best_el_crawl_clusters_list')
            # cluster_test(self.p, self.side_info, best_el_crawl_clusters_list, self.true_ent2clust, self.true_clust2ent, print_or_not=True)

            print('use_bert_update_seeds:', use_bert_update_seeds)
            if use_bert_update_seeds:
                if self.p.use_context and self.p.use_BERT:
                    fname1 = '../file/' + self.p.dataset + '/' + folder + '/BERT_fine-tune_label_' + str(self.training_time)
                    print('fname1:', fname1)
                    if not checkFile(fname1):
                        print('generate BERT_fine-tune_', self.training_time, fname1)
                        # self.label = best_el_crawl_clusters_list
                        self.label = el_prior_cluster_list
                        for i in range(20):
                            BERT_self_training_time = i
                            BM = BERT_Model(self.p, self.side_info, self.logger, clean_ent_list, self.label,
                                            self.true_ent2clust, self.true_clust2ent, self.training_time,
                                            BERT_self_training_time)
                            self.label = BM.fine_tune()
                        pickle.dump(self.label, open(fname1, 'wb'))
                    else:
                        print('load BERT_fine-tune_', self.training_time, fname1)
                        self.label = pickle.load(open(fname1, 'rb'))
                # old_label, new_label = best_el_crawl_clusters_list, self.label
                old_label, new_label = el_prior_cluster_list, self.label
            else:
                old_label, new_label = best_np_view_clusters_list, best_el_crawl_clusters_list

            print('old_label : ')
            cluster_test(self.p, self.side_info, old_label, self.true_ent2clust, self.true_clust2ent, print_or_not=True)
            print('new_label : ')
            cluster_test(self.p, self.side_info, new_label, self.true_ent2clust, self.true_clust2ent, print_or_not=True)
            exit()

            fname1, fname2 = '../file/' + self.p.dataset + '/' + folder + '/seed_trpIds_' + str(
                self.training_time), '../file/' + self.p.dataset + '/' + folder + '/seed_sim_' + str(self.training_time)
            if not checkFile(fname1) or not checkFile(fname2):
                print('generate new seeds ', self.training_time, fname1)
                # new_seed_pair_list = difference_cluster2pair(old_label, new_label, self.EL_seed)
                new_seed_pair_list = totol_cluster2pair(new_label)
                print('new_seed_pair_list:', type(new_seed_pair_list), len(new_seed_pair_list))
                self.new_seed_trpIds, self.new_seed_sim = pair2triples(new_seed_pair_list, self.side_info.ent_list, self.side_info.ent2id,
                                                                       self.side_info.id2ent, self.side_info.ent2triple_id_list,
                                                                       self.side_info.trpIds, self.ent2embed, cos_sim)

                pickle.dump(self.new_seed_trpIds, open(fname1, 'wb'))
                pickle.dump(self.new_seed_sim, open(fname2, 'wb'))
            else:
                print('load new seeds ', self.training_time, fname1)
                self.new_seed_trpIds = pickle.load(open(fname1, 'rb'))
                self.new_seed_sim = pickle.load(open(fname2, 'rb'))
            print('self.new_seed_trpIds:', type(self.new_seed_trpIds), len(self.new_seed_trpIds))
            print('self.new_seed_sim:', type(self.new_seed_sim), len(self.new_seed_sim))
            if self.generate:
                exit()
        exit()


        exit()