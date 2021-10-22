import gensim
from helper import *
# from DEC_keras_change_embedding import *
# from DEC_tf_keras_word_level import *
from Multi_view_co_training import Co_training, cluster2distribution
from Multi_view_self_training import Self_training
from DEC_multi_view import *
from Multi_view_spherical_kmeans import Multi_view_SphericalKMeans
from spherecluster import SphericalKMeans
import time

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
        self.ent2dict_embed = {}
        self.true_ent2clust, self.true_clust2ent = true_ent2clust, true_clust2ent

    def fit(self):
        clean_ent_list, clean_rel_list = [], []
        for ent in self.side_info.ent_list: clean_ent_list.append(ent.split('|')[0])
        for rel in self.side_info.rel_list: clean_rel_list.append(rel.split('|')[0])

        clean_ent_id_list, clean_rel_id_list = [], []
        for ent in clean_ent_list: clean_ent_id_list.append(self.side_info.ent2id[ent])
        for rel in clean_rel_list: clean_rel_id_list.append(self.side_info.rel2id[rel])
        print('clean_ent_list:', type(clean_ent_list), len(clean_ent_list))  # 19915
        print('clean_rel_list:', type(clean_rel_list), len(clean_rel_list))  # 18250

        print('use_Entity_linking_dict:', self.p.use_Entity_linking_dict)  # True
        if self.p.only_use_sub:
            self.sub_entity_linking_prior = {}
            if self.p.use_Entity_linking_dict:
                self.entity_linking_prior = self.side_info.ent_old_id2new_id
                clean_sub_list = []
                for sub_id, eid in enumerate(self.side_info.isSub.keys()):
                    clean_sub_list.append(clean_ent_list[eid])
                    if sub_id in self.sub_entity_linking_prior.keys():
                        continue
                    else:
                        self.sub_entity_linking_prior[sub_id] = self.entity_linking_prior[sub_id]
            else:
                clean_sub_list = []
                for sub_id, eid in enumerate(self.side_info.isSub.keys()):
                    clean_sub_list.append(clean_ent_list[eid])
            print('self.sub_entity_linking_prior:', type(self.sub_entity_linking_prior),
                  len(self.sub_entity_linking_prior))
            clean_ent_list = clean_sub_list

        self.entity_linking_prior = {}
        if self.p.use_Entity_linking_dict:
            self.entity_linking_prior = self.side_info.ent_old_id2new_id

        print('clean_ent_list:', type(clean_ent_list), len(clean_ent_list))
        print('self.entity_linking_prior:', type(self.entity_linking_prior), len(self.entity_linking_prior))  # 23735
        # el_prior_cluster_list = []
        # for i in range(len(clean_ent_list)):
        #     ent = clean_ent_list[i]
        #     old_id = self.side_info.ent2id[ent]
        #     new_id = self.entity_linking_prior[old_id]
        #     el_prior_cluster_list.append(new_id)
        # ave_prec, ave_recall, ave_f1, macro_prec, micro_prec, pair_prec, macro_recall, micro_recall, \
        # pair_recall, macro_f1, micro_f1, pair_f1, clusters, singletons = cluster_test(self.p, self.side_info,
        #                                                                               el_prior_cluster_list,
        #                                                                               self.true_ent2clust,
        #                                                                               self.true_clust2ent)
        # print('only use el prior dict ... ')
        # print('Ave-prec=', ave_prec, 'macro_prec=', macro_prec, 'micro_prec=', micro_prec,
        #       'pair_prec=', pair_prec)
        # print('Ave-recall=', ave_recall, 'macro_recall=', macro_recall, 'micro_recall=', micro_recall,
        #       'pair_recall=', pair_recall)
        # print('Ave-F1=', ave_f1, 'macro_f1=', macro_f1, 'micro_f1=', micro_f1, 'pair_f1=', pair_f1)
        # print('clusters=', clusters, 'singletons=', singletons)
        # exit()

        sentence_list = []  # need to add the sentence_list after EL dict
        for ent in clean_ent_list:
            id = self.side_info.ent2id[ent]
            sentences_str = str()
            if self.p.only_use_sub:
                if id in self.side_info.isSub.keys():  # only use subject's sentence
                    # if self.p.use_Entity_linking_dict:
                    #     id = self.side_info.ent_old_id2new_id[id]
                    sentences_ids = self.side_info.ent_id2sentence_list[id]
                    if self.p.sentence_view_use_first_sentence:
                        sentences_id = sentences_ids[0]
                        sentences_str = self.side_info.sentence_List[sentences_id]
                    else:
                        for sentences_id in sentences_ids:
                            sentence = self.side_info.sentence_List[sentences_id] + ' '
                            sentences_str += sentence
                    sentence_list.append(sentences_str)
            else:
                # if self.p.use_Entity_linking_dict:
                #     id = self.side_info.ent_old_id2new_id[id]
                sentences_ids = self.side_info.ent_id2sentence_list[id]
                if self.p.sentence_view_use_first_sentence:
                    sentences_id = sentences_ids[0]
                    sentences_str = self.side_info.sentence_List[sentences_id]
                else:
                    for sentences_id in sentences_ids:
                        sentence = self.side_info.sentence_List[sentences_id] + ' '
                        sentences_str += sentence
                sentence_list.append(sentences_str)

        print('sentence_list:', type(sentence_list), len(sentence_list))

        true_answer_list = []
        for i in clean_ent_list:
            true_answer_list.append(i)
        clust_id = [self.true_clust2ent.keys()]
        clust_id = list(set(clust_id[0]))

        for clust in self.true_clust2ent.keys():
            answer_id = clust_id.index(clust)
            ent_id_list = list(self.true_clust2ent[clust])
            for ent_id in ent_id_list:
                ent = ent_id.split('|')[0]
                ent_id = self.side_info.ent2id[ent]
                true_answer_list[ent_id] = answer_id

        print('self.p.sentence_view_use_first_sentence:', self.p.sentence_view_use_first_sentence)
        if self.p.sentence_view_use_first_sentence:
            sentence_tag = '_first_sentence'
        else:
            sentence_tag = '_all_sentence'
        ''' Intialize embeddings '''
        fname1, fname2, fname3 = './file/1E_init', './file/1R_init', './file/1S_init' + sentence_tag
        fname4, fname5 = './file/1E_pre_trained_word_embed', './file/1R_pre_trained_word_embed'
        fname6, fname7 = './file/1E_word_list', './file/1R_word_list'
        fname8, fname9, fname10 = './file/1E_word_index_list', './file/1R_word_index_list', './file/1S_word_index_list'
        if not checkFile(fname1) or not checkFile(fname2) or not checkFile(fname3) or not checkFile(fname4):
            print('generate init word2vec dict embeddings')
            model = gensim.models.KeyedVectors.load_word2vec_format(self.p.embed_loc, binary=False)
            if self.p.embed_init == 'crawl':
                print('clean_ent_list:', type(clean_ent_list), len(clean_ent_list))
                self.E_init = getEmbeddings(model, clean_ent_list, self.p.embed_dims)
                self.R_init = getEmbeddings(model, clean_rel_list, self.p.embed_dims)
                self.S_init = getEmbeddings(model, sentence_list, self.p.embed_dims)
            else:
                print('generate init random embeddings')
                self.E_init = np.random.rand(len(clean_ent_list), self.p.embed_dims)
                self.R_init = np.random.rand(len(clean_rel_list), self.p.embed_dims)
                self.S_init = np.random.rand(len(sentence_list), self.p.embed_dims)

            self.pre_trained_entity_word_embed, self.entity_and_sentence_word_list, self.np_word_index_list\
                , self.sentence_word_index_list = get_multi_view_word_embeddings(model, clean_ent_list, sentence_list, self.p.embed_dims)
            # self.pre_trained_entity_word_embed is the word (np and sentence) embedding in crawl dict
            # self.entity_and_sentence_word_list is the word (np and sentence) list

            # self.pre_trained_entity_word_embed, self.entity_word_list, self.entity_word_index_list = \
            #     get_word_embeddings(model, clean_ent_list, self.p.embed_dims)
            self.pre_trained_relation_word_embed, self.relation_word_list, self.relation_word_index_list = \
                get_word_embeddings(model, clean_rel_list, self.p.embed_dims)

            pickle.dump(self.E_init, open(fname1, 'wb'))
            pickle.dump(self.R_init, open(fname2, 'wb'))
            pickle.dump(self.S_init, open(fname3, 'wb'))
            pickle.dump(self.pre_trained_entity_word_embed, open(fname4, 'wb'))
            pickle.dump(self.pre_trained_relation_word_embed, open(fname5, 'wb'))
            pickle.dump(self.entity_and_sentence_word_list, open(fname6, 'wb'))
            pickle.dump(self.relation_word_list, open(fname7, 'wb'))
            pickle.dump(self.np_word_index_list, open(fname8, 'wb'))
            pickle.dump(self.relation_word_index_list, open(fname9, 'wb'))
            pickle.dump(self.sentence_word_index_list, open(fname10, 'wb'))
        else:
            print('load init crawl pre-trained-embeddings')
            self.E_init = pickle.load(open(fname1, 'rb'))
            self.R_init = pickle.load(open(fname2, 'rb'))
            self.S_init = pickle.load(open(fname3, 'rb'))
            self.pre_trained_entity_word_embed = pickle.load(open(fname4, 'rb'))
            self.pre_trained_relation_word_embed = pickle.load(open(fname5, 'rb'))
            self.entity_and_sentence_word_list = pickle.load(open(fname6, 'rb'))
            self.relation_word_list = pickle.load(open(fname7, 'rb'))
            self.np_word_index_list = pickle.load(open(fname8, 'rb'))
            self.relation_word_index_list = pickle.load(open(fname9, 'rb'))
            self.sentence_word_index_list = pickle.load(open(fname10, 'rb'))
        print('self.E_init:', type(self.E_init), len(self.E_init))  # <class 'numpy.ndarray'> 23735
        print('self.R_init:', type(self.R_init), len(self.R_init))  # <class 'numpy.ndarray'> 18288
        print('self.S_init:', type(self.S_init), len(self.S_init))  # <class 'numpy.ndarray'> 23735
        print('self.pre_trained_entity_word_embed:', type(self.pre_trained_entity_word_embed), self.pre_trained_entity_word_embed.shape)  # <class 'numpy.ndarray'> (22737, 300)
        print('self.pre_trained_relation_word_embed:', type(self.pre_trained_relation_word_embed), self.pre_trained_relation_word_embed.shape)  # <class 'numpy.ndarray'> (6147, 300)
        print('self.entity_and_sentence_word_list:', type(self.entity_and_sentence_word_list), len(self.entity_and_sentence_word_list))  # <class 'list'> 22737
        print('self.relation_word_list:', type(self.relation_word_list), len(self.relation_word_list))  # <class 'list'> 6147
        print('self.np_word_index_list:', type(self.np_word_index_list), len(self.np_word_index_list))  # <class 'list'> 23735
        print('self.relation_word_index_list:', type(self.relation_word_index_list), len(self.relation_word_index_list))  # <class 'list'> 18288
        print('self.sentence_word_index_list:', type(self.sentence_word_index_list), len(self.sentence_word_index_list))  # <class 'list'> 23735

        # two_dim_matrix_one = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0]])
        # two_dim_matrix_two = np.array([[0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 1]])
        #
        # two_multi_res = np.dot(two_dim_matrix_one, two_dim_matrix_two)
        # print('two_multi_res: %s' % (two_multi_res))
        # two_multi_res = np.dot(two_dim_matrix_two, two_dim_matrix_one)
        # print('two_multi_res: %s' % (two_multi_res))
        # exit()
        fname1, fname2 = './file/el_prior_cluster_list', './file/np_view_clusters_list'
        if not checkFile(fname1) or not checkFile(fname2):
            print('generate view_1 and view_2')
            el_prior_cluster_list = []
            el_repeat_old_dict = dict()
            for i in range(len(clean_ent_list)):
                ent = clean_ent_list[i]
                old_id = self.side_info.ent2id[ent]
                new_id = self.entity_linking_prior[old_id]
                el_prior_cluster_list.append(new_id)
                if new_id not in el_repeat_old_dict:
                    el_repeat_old_dict.update({new_id: 1})

            self.cluster_threshold = 0.33
            np_view_clusters, np_view_clusters_center = HAC_getClusters(self.p, self.E_init, self.cluster_threshold)
            np_view_clusters = list(np_view_clusters)
            np_view_clusters_list = []
            el_repeat_dict = dict()
            np_result_dict = dict()
            num = 0
            for i in range(len(np_view_clusters)):
                np_id = np_view_clusters[i]
                el_id = el_prior_cluster_list[i]
                if np_id in np_result_dict:
                    np_view_clusters_list.append(np_result_dict[np_id])
                else:
                    if el_id in el_repeat_dict:
                        while num in np_view_clusters_list or num in el_repeat_old_dict:
                            num += 1
                        np_result_dict.update({np_id: num})
                        np_view_clusters_list.append(np_result_dict[np_id])
                        num += 1
                    else:
                        np_view_clusters_list.append(el_id)
                        np_result_dict.update({np_id: el_id})
                        el_repeat_dict.update({el_id: 1})
                if el_id not in el_repeat_dict:
                    el_repeat_dict.update({el_id: 1})

            pickle.dump(el_prior_cluster_list, open(fname1, 'wb'))
            pickle.dump(np_view_clusters_list, open(fname2, 'wb'))
        else:
            print('load view_1 and view_2')
            el_prior_cluster_list = pickle.load(open(fname1, 'rb'))
            np_view_clusters_list = pickle.load(open(fname2, 'rb'))

        use_views_num = self.p.use_views_num
        print('use_views_num:', use_views_num)

        ave_prec, ave_recall, ave_f1, macro_prec, micro_prec, pair_prec, macro_recall, micro_recall, \
        pair_recall, macro_f1, micro_f1, pair_f1, clusters, singletons = cluster_test(self.p, self.side_info,
                                                                                      el_prior_cluster_list,
                                                                                      self.true_ent2clust,
                                                                                      self.true_clust2ent)
        print('only use el prior dict ... ')
        print('el_prior_cluster_list:', type(el_prior_cluster_list), len(el_prior_cluster_list),
              el_prior_cluster_list[0:100])
        print('Ave-prec=', ave_prec, 'macro_prec=', macro_prec, 'micro_prec=', micro_prec,
              'pair_prec=', pair_prec)
        print('Ave-recall=', ave_recall, 'macro_recall=', macro_recall, 'micro_recall=', micro_recall,
              'pair_recall=', pair_recall)
        print('Ave-F1=', ave_f1, 'macro_f1=', macro_f1, 'micro_f1=', micro_f1, 'pair_f1=', pair_f1)
        print('clusters=', clusters, 'singletons=', singletons)
        print()

        ave_prec, ave_recall, ave_f1, macro_prec, micro_prec, pair_prec, macro_recall, micro_recall, \
        pair_recall, macro_f1, micro_f1, pair_f1, clusters, singletons = cluster_test(self.p, self.side_info,
                                                                                      np_view_clusters_list,
                                                                                      self.true_ent2clust,
                                                                                      self.true_clust2ent)
        print('only use crawl ... ')
        print('np_view_clusters_list:', type(np_view_clusters_list), len(np_view_clusters_list),
              np_view_clusters_list[0:100])
        print('Ave-prec=', ave_prec, 'macro_prec=', macro_prec, 'micro_prec=', micro_prec,
              'pair_prec=', pair_prec)
        print('Ave-recall=', ave_recall, 'macro_recall=', macro_recall, 'micro_recall=', micro_recall,
              'pair_recall=', pair_recall)
        print('Ave-F1=', ave_f1, 'macro_f1=', macro_f1, 'micro_f1=', micro_f1, 'pair_f1=', pair_f1)
        print('clusters=', clusters, 'singletons=', singletons)
        print()

        Test_training_mothod = 'self_training'
        if Test_training_mothod == 'self_training':
            view_1 = np.array(np_view_clusters_list)
            view_2 = np.array(el_prior_cluster_list)
            view_1 = view_1.reshape((len(np_view_clusters_list), 1))
            view_2 = view_2.reshape((len(np_view_clusters_list), 1))

            if use_views_num == 4:
                crawl_el_cluster_list = []
                for i in range(len(np_view_clusters_list)):
                    old_id = np_view_clusters_list[i]
                    new_id = el_prior_cluster_list[old_id]
                    crawl_el_cluster_list.append(new_id)
                print('crawl_el_cluster_list:', type(crawl_el_cluster_list), len(crawl_el_cluster_list))
                ave_prec, ave_recall, ave_f1, macro_prec, micro_prec, pair_prec, macro_recall, micro_recall, \
                pair_recall, macro_f1, micro_f1, pair_f1, clusters, singletons = cluster_test(self.p, self.side_info,
                                                                                              crawl_el_cluster_list,
                                                                                              self.true_ent2clust,
                                                                                              self.true_clust2ent)
                print('crawl + el ... ')
                print('Ave-prec=', ave_prec, 'macro_prec=', macro_prec, 'micro_prec=', micro_prec,
                      'pair_prec=', pair_prec)
                print('Ave-recall=', ave_recall, 'macro_recall=', macro_recall, 'micro_recall=', micro_recall,
                      'pair_recall=', pair_recall)
                print('Ave-F1=', ave_f1, 'macro_f1=', macro_f1, 'micro_f1=', micro_f1, 'pair_f1=', pair_f1)
                print('clusters=', clusters, 'singletons=', singletons)
                print()

                el_crawl_cluster_list = []
                for i in range(len(el_prior_cluster_list)):
                    old_id = el_prior_cluster_list[i]
                    new_id = np_view_clusters_list[old_id]
                    el_crawl_cluster_list.append(new_id)
                print('el_crawl_cluster_list:', type(el_crawl_cluster_list), len(el_crawl_cluster_list))
                ave_prec, ave_recall, ave_f1, macro_prec, micro_prec, pair_prec, macro_recall, micro_recall, \
                pair_recall, macro_f1, micro_f1, pair_f1, clusters, singletons = cluster_test(self.p, self.side_info,
                                                                                              el_crawl_cluster_list,
                                                                                              self.true_ent2clust,
                                                                                              self.true_clust2ent)
                print('el + crawl ... ')
                print('Ave-prec=', ave_prec, 'macro_prec=', macro_prec, 'micro_prec=', micro_prec,
                      'pair_prec=', pair_prec)
                print('Ave-recall=', ave_recall, 'macro_recall=', macro_recall, 'micro_recall=', micro_recall,
                      'pair_recall=', pair_recall)
                print('Ave-F1=', ave_f1, 'macro_f1=', macro_f1, 'micro_f1=', micro_f1, 'pair_f1=', pair_f1)
                print('clusters=', clusters, 'singletons=', singletons)
                print()
                view_3 = np.array(crawl_el_cluster_list)
                view_4 = np.array(el_crawl_cluster_list)
                view_3 = view_3.reshape((len(np_view_clusters_list), 1))
                view_4 = view_4.reshape((len(np_view_clusters_list), 1))
                input_views = [view_1, view_2, view_3, view_4]

                from cluster_f1_test import fake_true_cluster_test
                test_dataset_c_input, test_dataset_e_input, test_dataset_macro_precision_input = [], [], []
                # # el_prior_cluster_list  # view_2
                # # crawl_el_cluster_list  # view_3
                # # np_view_clusters_list  # view_1
                # # el_crawl_cluster_list  # view_4
                # print('0.7415 to 0.7483')
                # fake_true_cluster_test(params=self.p, side_info=self.side_info, c_cluster_predict_list=el_prior_cluster_list, e_cluster_predict_list=crawl_el_cluster_list)
                print('0.7415 to 0.7739')
                macro_prec = fake_true_cluster_test(params=self.p, side_info=self.side_info, c_cluster_predict_list=el_prior_cluster_list, e_cluster_predict_list=np_view_clusters_list)
                test_dataset_c_input.append(view_2)
                test_dataset_e_input.append(view_1)
                test_dataset_macro_precision_input.append(macro_prec)
                # print('0.7415 to 0.8148')
                # fake_true_cluster_test(params=self.p, side_info=self.side_info, c_cluster_predict_list=el_prior_cluster_list, e_cluster_predict_list=el_crawl_cluster_list)
                #
                # print('0.7483 to 0.7415')
                # fake_true_cluster_test(params=self.p, side_info=self.side_info, c_cluster_predict_list=crawl_el_cluster_list, e_cluster_predict_list=el_prior_cluster_list)
                print('0.7483 to 0.7739')
                macro_prec = fake_true_cluster_test(params=self.p, side_info=self.side_info, c_cluster_predict_list=crawl_el_cluster_list, e_cluster_predict_list=np_view_clusters_list)
                test_dataset_c_input.append(view_3)
                test_dataset_e_input.append(view_1)
                test_dataset_macro_precision_input.append(macro_prec)
                # print('0.7483 to 0.8148')
                # fake_true_cluster_test(params=self.p, side_info=self.side_info, c_cluster_predict_list=crawl_el_cluster_list, e_cluster_predict_list=el_crawl_cluster_list)
                #
                print('0.7739 to 0.7415')
                macro_prec = fake_true_cluster_test(params=self.p, side_info=self.side_info, c_cluster_predict_list=np_view_clusters_list, e_cluster_predict_list=el_prior_cluster_list)
                test_dataset_c_input.append(view_1)
                test_dataset_e_input.append(view_2)
                test_dataset_macro_precision_input.append(macro_prec)
                print('0.7739 to 0.7483')
                macro_prec = fake_true_cluster_test(params=self.p, side_info=self.side_info, c_cluster_predict_list=np_view_clusters_list, e_cluster_predict_list=crawl_el_cluster_list)
                test_dataset_c_input.append(view_1)
                test_dataset_e_input.append(view_3)
                test_dataset_macro_precision_input.append(macro_prec)
                print('0.7739 to 0.8148')
                macro_prec = fake_true_cluster_test(params=self.p, side_info=self.side_info, c_cluster_predict_list=np_view_clusters_list, e_cluster_predict_list=el_crawl_cluster_list)
                test_dataset_c_input.append(view_1)
                test_dataset_e_input.append(view_4)
                test_dataset_macro_precision_input.append(macro_prec)
                #
                # print('0.8148 to 0.7415')
                # fake_true_cluster_test(params=self.p, side_info=self.side_info, c_cluster_predict_list=el_crawl_cluster_list, e_cluster_predict_list=el_prior_cluster_list)
                # print('0.8148 to 0.7483')
                # fake_true_cluster_test(params=self.p, side_info=self.side_info, c_cluster_predict_list=el_crawl_cluster_list, e_cluster_predict_list=crawl_el_cluster_list)
                print('0.8148 to 0.7739')
                macro_prec = fake_true_cluster_test(params=self.p, side_info=self.side_info, c_cluster_predict_list=el_crawl_cluster_list, e_cluster_predict_list=np_view_clusters_list)
                test_dataset_c_input.append(view_4)
                test_dataset_e_input.append(view_1)
                test_dataset_macro_precision_input.append(macro_prec)
                # exit()
            elif use_views_num == 2:
                input_views = [view_1, view_2]
            else:
                input_views = [view_1]
            t0 = time.time()

            st = Self_training(params=self.p, side_info=self.side_info, input=input_views, target=view_4,
                               true_ent2clust=self.true_ent2clust, true_clust2ent=self.true_clust2ent,
                               true_answer=true_answer_list)
            cluster_predict_list, all_f1_list, loss_list = st.fit()
            print('final result : ')
            print('cluster_predict_list:', type(cluster_predict_list), len(cluster_predict_list), cluster_predict_list)
            ave_prec, ave_recall, ave_f1, macro_prec, micro_prec, pair_prec, macro_recall, micro_recall, \
            pair_recall, macro_f1, micro_f1, pair_f1, clusters, singletons = cluster_test(self.p, self.side_info,
                                                                                          cluster_predict_list,
                                                                                          self.true_ent2clust,
                                                                                          self.true_clust2ent)
            print('Ave-prec=', ave_prec, 'macro_prec=', macro_prec, 'micro_prec=', micro_prec,
                  'pair_prec=', pair_prec)
            print('Ave-recall=', ave_recall, 'macro_recall=', macro_recall, 'micro_recall=', micro_recall,
                  'pair_recall=', pair_recall)
            print('Ave-F1=', ave_f1, 'macro_f1=', macro_f1, 'micro_f1=', micro_f1, 'pair_f1=', pair_f1)
            print('clusters=', clusters, 'singletons=', singletons)
            time_cost = time.time() - t0
            print('clustering time: ', time_cost / 60, 'minute')
            print()
            import matplotlib.pyplot as plt
            plt.plot(all_f1_list)
            plt.ylabel('Ave-F1')
            plt.show()
            plt.plot(loss_list)
            plt.ylabel('Loss')
            plt.show()
            exit()
        else:
            view_1 = cluster2distribution(np_view_clusters_list)
            view_2 = cluster2distribution(el_prior_cluster_list)
            t0 = time.time()
            ct = Co_training(params=self.p, side_info=self.side_info, view_1=view_1, view_2=view_2,
                             true_ent2clust=self.true_ent2clust, true_clust2ent=self.true_clust2ent,
                             true_answer=true_answer_list)
            cluster_predict_list, ave_f1_list, loss_list = ct.fit()
            print('final result : ')
            print('cluster_predict_list:', type(cluster_predict_list), len(cluster_predict_list), cluster_predict_list)
            ave_prec, ave_recall, ave_f1, macro_prec, micro_prec, pair_prec, macro_recall, micro_recall, \
            pair_recall, macro_f1, micro_f1, pair_f1, clusters, singletons = cluster_test(self.p, self.side_info,
                                                                                          cluster_predict_list,
                                                                                          self.true_ent2clust,
                                                                                          self.true_clust2ent)
            print('Ave-prec=', ave_prec, 'macro_prec=', macro_prec, 'micro_prec=', micro_prec,
                  'pair_prec=', pair_prec)
            print('Ave-recall=', ave_recall, 'macro_recall=', macro_recall, 'micro_recall=', micro_recall,
                  'pair_recall=', pair_recall)
            print('Ave-F1=', ave_f1, 'macro_f1=', macro_f1, 'micro_f1=', micro_f1, 'pair_f1=', pair_f1)
            print('clusters=', clusters, 'singletons=', singletons)
            time_cost = time.time() - t0
            print('clustering time: ', time_cost / 60, 'minute')
            print()
        exit()

        # print('entity view : ')
        # self.v_init = self.E_init
        # embed2f1(params=self.p, embed=self.v_init,
        #          cluster_threshold_real=self.cluster_threshold, side_info=self.side_info,
        #          true_ent2clust=self.true_ent2clust, true_clust2ent=self.true_clust2ent)
        # print()
        #
        # self.v_init = (self.E_init + self.S_init) / 2
        # self.v_init = self.S_init
        # print('context view : ')
        # embed2f1(params=self.p, embed=self.v_init,
        #          cluster_threshold_real=self.cluster_threshold, side_info=self.side_info,
        #          true_ent2clust=self.true_ent2clust, true_clust2ent=self.true_clust2ent)
        # print()

        if self.p.Multi_view_choice == 'spherical-k-means':
            print('Model is multi-view spherical-k-means')
            t0 = time.time()
            # n_cluster = len(self.E_init)
            n_cluster = 13053
            # n_cluster = 11500 -- 13500
            mv_skm = Multi_view_SphericalKMeans(n_clusters=n_cluster, init='k-means++', n_init=1, max_iter=100,
                                                n_jobs=-1, verbose=1, p=self.p, side_info=self.side_info,
                                                true_ent2clust=self.true_ent2clust, true_clust2ent=self.true_clust2ent)
            print("Clustering with %s" % mv_skm)

            # mv_skm.fit(self.E_init, self.V_init)
            mv_skm.fit((self.E_init + self.S_init) / 2, self.E_init)
            print('n_cluster:', type(n_cluster), n_cluster)
            cluster_predict_list = mv_skm.labels_
            print()
            print('final result : ')
            print('cluster_predict_list:', type(cluster_predict_list), len(cluster_predict_list), cluster_predict_list)
            ave_prec, ave_recall, ave_f1, macro_prec, micro_prec, pair_prec, macro_recall, micro_recall, \
            pair_recall, macro_f1, micro_f1, pair_f1, clusters, singletons = cluster_test(self.p, self.side_info,
                                                                                          cluster_predict_list,
                                                                                          self.true_ent2clust,
                                                                                          self.true_clust2ent)
            print('Ave-prec=', ave_prec, 'macro_prec=', macro_prec, 'micro_prec=', micro_prec,
                  'pair_prec=', pair_prec)
            print('Ave-recall=', ave_recall, 'macro_recall=', macro_recall, 'micro_recall=', micro_recall,
                  'pair_recall=', pair_recall)
            print('Ave-F1=', ave_f1, 'macro_f1=', macro_f1, 'micro_f1=', micro_f1, 'pair_f1=', pair_f1)
            print('clusters=', clusters, 'singletons=', singletons)
            time_cost = time.time() - t0
            print('clustering time: ', time_cost / 60, 'minute')
            print()
            exit()
        else:
            print('Model is multi-view DEC')
            # n_clusters = self.p.n_clusters
            # n_clusters = 1000
            self.np_word_index_matrix = np.zeros((len(self.np_word_index_list),
                                                  len(self.entity_and_sentence_word_list)), dtype=np.float32)
            self.sentence_word_index_matrix = np.zeros((len(self.sentence_word_index_list),
                                                        len(self.entity_and_sentence_word_list)), dtype=np.float32)
            for i in range(len(self.np_word_index_list)):  # i is the i's NP
                word_index = self.np_word_index_list[i]  # NP's word index list
                num = 1 / len(word_index)  # after get average
                for j in range(len(word_index)):
                    index = word_index[j]  # index is every NP's word index
                    self.np_word_index_matrix[i, index] = num
            print('self.np_word_index_matrix:', type(self.np_word_index_matrix), self.np_word_index_matrix.shape)

            for i in range(len(self.sentence_word_index_list)):  # i is the i's sentence
                word_index = self.sentence_word_index_list[i]  # sentence's word index list
                num = 1 / len(word_index)  # after get average
                for j in range(len(word_index)):
                    index = word_index[j]  # index is every sentence's word index
                    self.sentence_word_index_matrix[i, index] = num
            print('self.sentence_word_index_matrix:', type(self.sentence_word_index_matrix),
                  self.sentence_word_index_matrix.shape)

            dec = DEC_multi_view(params=self.p, side_info=self.side_info, word_embed=self.pre_trained_entity_word_embed,
                                 np_index=self.np_word_index_matrix, np_view_embed=self.E_init,
                                 context_index=self.sentence_word_index_matrix,
                                 context_view_embed=(self.E_init + self.S_init) / 2,
                                 # context_view_embed=self.S_init,
                                 true_ent2clust=self.true_ent2clust, true_clust2ent=self.true_clust2ent,
                                 el_prior=self.entity_linking_prior, true_answer=true_answer_list)
            cluster_predict = dec.y_pred
            all_f1_list = dec.all_f1_list

            print('cluster_predict:', type(cluster_predict), cluster_predict)
            cluster_predict_list = list(cluster_predict)
            print('cluster_predict_list:', type(cluster_predict_list), len(cluster_predict_list), cluster_predict_list)
            # exit()
            f = open('cluster_predict_list.csv', 'w')
            for predict in cluster_predict_list:
                f.write(str(predict))
                f.write('\n')
            f.close()
            print('cluster_predict_list is saved !')
            import matplotlib.pyplot as plt
            plt.plot(all_f1_list)
            plt.ylabel('Ave-F1')
            plt.show()

        # if self.p.use_Entity_linking_dict:
        if not self.p.use_Entity_linking_dict:

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
