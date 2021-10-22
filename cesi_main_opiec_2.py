from helper import *
import gensim
from sideInfo_param_tihuan_wutf import SideInfo  # For processing data and side information
# from sideInfo_without_assume import SideInfo  # For processing data and side information
# from sideInfo_HAN import SideInfo
# from embeddings_RotatE_max_margin_test import Embeddings  # For learning embeddings
from embeddings_multi_view import Embeddings
# from cluster_change import Clustering  # For clustering learned embeddings
# from cluster import Clustering  # For clustering learned embeddings
from cluster_without_assume import Clustering  # For clustering learned embeddings
#from cluster_check import Clustering  # For clustering learned embeddings
from metrics import evaluate  # Evaluation metrics
from utils import *
# reload(sys);
# sys.setdefaultencoding('utf-8')			# Swtching from ASCII to UTF-8 encoding
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
''' *************************************** DATASET PREPROCESSING **************************************** '''


class CESI_Main(object):

    def __init__(self, args):
        self.p = args
        self.logger = getLogger(args.name, args.log_dir, args.config_dir)
        self.logger.info('Running {}'.format(args.name))
        self.read_triples()


    def read_triples(self):
        self.logger.info('Reading Triples')

        fname = self.p.out_path + self.p.file_triples  # File for storing processed triples
        self.triples_list = []  # List of all triples in the dataset
        self.amb_ent = ddict(int)  # Contains ambiguous entities in the dataset
        self.amb_mentions = {}  # Contains all ambiguous mentions
        self.isAcronym = {}  # Contains all mentions which can be acronyms

        print('dataset:', args.dataset)
        if args.dataset == 'OPIEC59k':
            print('load OPIEC_dataset ... ')
            self.triples_list = pickle.load(open(args.data_path, 'rb'))

            ''' Ground truth clustering '''
            self.true_ent2clust = ddict(set)
            for trp in self.triples_list:
                sub_u = trp['triple_unique'][0]
                # self.true_ent2clust[sub_u].add(trp['true_sub_link'])
                self.true_ent2clust[sub_u].add(trp['subject_wiki_link'])
                # obj_u = trp['triple_unique'][2]
                # self.true_ent2clust[obj_u].add(trp['true_obj_link'])
            self.true_clust2ent = invertDic(self.true_ent2clust, 'm2os')

        else:
            if not checkFile(fname):
                with codecs.open(args.data_path, encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        trp = json.loads(line.strip())

                        trp['raw_triple'] = trp['triple']
                        sub, rel, obj = map(str, trp['triple'])

                        '''
                        if sub.isalpha() and sub.isupper(): self.isAcronym[
                            proc_ent(sub)] = 1  # Check if the subject is an acronym
                        if obj.isalpha() and obj.isupper(): self.isAcronym[
                            proc_ent(obj)] = 1  # Check if the object  is an acronym

                        sub, rel, obj = proc_ent(sub), trp['triple_norm'][1], proc_ent(
                            obj)  # Get Morphologically normalized subject, relation, object

                        # for reverb45k_test_new dataset
                        sub, rel, obj = str(sub).lower(), str(rel).lower(), str(obj).lower()
                        '''

                        if len(sub) == 0 or len(rel) == 0 or len(obj) == 0: continue  # Ignore incomplete triples

                        trp['triple'] = [sub, rel, obj]
                        trp['triple_unique'] = [sub + '|' + str(trp['_id']), rel + '|' + str(trp['_id']),
                                                obj + '|' + str(trp['_id'])]
                        trp['ent_lnk_sub'] = trp['entity_linking']['subject']
                        trp['ent_lnk_obj'] = trp['entity_linking']['object']
                        trp['true_sub_link'] = trp['true_link']['subject']
                        trp['true_obj_link'] = trp['true_link']['object']
                        trp['rel_info'] = trp['kbp_info']  # KBP side info for relation

                        self.triples_list.append(trp)

                with open(fname, 'w') as f:
                    f.write('\n'.join([json.dumps(triple) for triple in self.triples_list]))
                    self.logger.info('\tCached triples')
            else:
                self.logger.info('\tLoading cached triples')
                with open(fname) as f:
                    self.triples_list = [json.loads(triple) for triple in f.read().split('\n')]

            ''' Ground truth clustering '''
            self.true_ent2clust = ddict(set)
            for trp in self.triples_list:
                sub_u = trp['triple_unique'][0]
                self.true_ent2clust[sub_u].add(trp['true_sub_link'])
                # obj_u = trp['triple_unique'][2]
                # self.true_ent2clust[obj_u].add(trp['true_obj_link'])
            self.true_clust2ent = invertDic(self.true_ent2clust, 'm2os')

        ''' Identifying ambiguous entities '''
        amb_clust = {}
        for trp in self.triples_list:
            sub = trp['triple'][0]
            for tok in sub.split():
                amb_clust[tok] = amb_clust.get(tok, set())
                amb_clust[tok].add(sub)

        for rep, clust in amb_clust.items():
            if rep in clust and len(clust) >= 3:
                self.amb_ent[rep] = len(clust)
                for ele in clust: self.amb_mentions[ele] = 1

        print('self.triples_list:', type(self.triples_list), len(self.triples_list))
        print('self.true_clust2ent:', len(self.true_clust2ent))
        print('self.true_ent2clust:', len(self.true_ent2clust))

    def get_sideInfo(self):
        self.logger.info('Side Information Acquisition')
        fname = self.p.out_path + self.p.file_sideinfo_pkl
        if self.p.use_assume:
            self.logger.info('use assume...')
        else:
            self.logger.info('do not use assume...')


        if not checkFile(fname):
            self.side_info = SideInfo(self.p, self.triples_list)
            # self.logger.info('\tName Side info');
            # self.side_info.nameEmbedding()  # KBP side information

            del self.side_info.file
            pickle.dump(self.side_info, open(fname, 'wb'))
            self.logger.info('\tCached Side Information')
        else:
            self.logger.info('\tLoading cached Side Information')
            self.side_info = pickle.load(open(fname, 'rb'))

    def embedKG(self):
        self.logger.info("Embedding NP and relation phrases")

        fname1 = self.p.out_path + self.p.file_entEmbed
        fname2 = self.p.out_path + self.p.file_relEmbed

        if not checkFile(fname1) or not checkFile(fname2):
            embed = Embeddings(self.p, self.side_info, self.logger, true_ent2clust=self.true_ent2clust,
                               true_clust2ent=self.true_clust2ent, triple_list=self.triples_list)
            embed.fit()

            self.ent2embed = embed.ent2embed  # Get the learned NP embeddings
            self.rel2embed = embed.rel2embed  # Get the learned RP embeddings

            pickle.dump(self.ent2embed, open(fname1, 'wb'))
            pickle.dump(self.rel2embed, open(fname2, 'wb'))
        else:
            self.logger.info('\tLoading cached Embeddings')
            self.ent2embed = pickle.load(open(fname1, 'rb'))
            self.rel2embed = pickle.load(open(fname2, 'rb'))

    #def cluster(self, cluster_threshold_real, weight_real):
    def cluster(self, cluster_threshold_real):
        self.logger.info('Clustering NPs and relation phrases')

        fname1 = self.p.out_path + self.p.file_entClust
        fname2 = self.p.out_path + self.p.file_relClust

        '''
        # new code for cluster while change the threshold
        self.sub2embed, self.sub2id = {}, {}  # Clustering only subjects
        self.sub2name_embed = {}
        for sub_id, eid in enumerate(self.isSub.keys()):
            self.sub2id[eid] = sub_id
            self.sub2embed[sub_id] = self.ent2embed[eid]
            # new_id = self.side_info.new_ent2id[self.side_info.id2ent[self.side_info.ent_old_id2new_id[eid]]]
            # self.sub2name_embed[sub_id] = self.side_info.ent2embed_name[new_id]
        self.id2sub = invertDic(self.sub2id)
        clust = Clustering(self.sub2embed, self.ent_freq, self.id2sub, self.p, cluster_threshold_real)
        self.ent_clust = clust.ent_clust
        '''


        #'''
        # new code for cluster while change the threshold
        # only sub
        self.sub2embed, self.sub2id = {}, {}  # Clustering only subjects
        for sub_id, eid in enumerate(self.side_info.isSub.keys()):
            self.sub2id[eid] = sub_id
            # new_id = self.side_info.new_ent2id[self.side_info.id2ent[self.side_info.ent_old_id2new_id[eid]]]
            # self.sub2embed[sub_id] = self.E_init[new_id]
            self.sub2embed[sub_id] = self.ent2embed[eid]
        self.side_info.id2sub = invertDic(self.sub2id)

        clust = Clustering(self.sub2embed, self.rel2embed, self.side_info, self.p, cluster_threshold_real, issub=True)
        self.ent_clust = clust.ent_clust
        #self.rel_clust = clust.rel_clust
        #'''

        '''
        if not checkFile(fname1):
        #if not checkFile(fname1) or not checkFile(fname2):

            self.sub2embed, self.sub2id = {}, {}  # Clustering only subjects
            self.sub2name_embed = {}
            for sub_id, eid in enumerate(self.side_info.isSub.keys()):
                self.sub2id[eid] = sub_id
                self.sub2embed[sub_id] = self.ent2embed[eid]
                #self.sub2name_embed[sub_id] = self.side_info.ent2embed_name[eid]
            self.side_info.id2sub = invertDic(self.sub2id)

            #clust = Clustering(self.sub2embed, self.rel2embed, self.side_info, self.p, self.sub2name_embed,
                               #self.side_info.rel2embed_name, cluster_threshold_real, weight_real)
            clust = Clustering(self.sub2embed, self.rel2embed, self.side_info, self.p, cluster_threshold_real)
            self.ent_clust = clust.ent_clust
            #self.rel_clust = clust.rel_clust

            dumpCluster(fname1, self.ent_clust, self.side_info.id2ent)
            #dumpCluster(fname2, self.rel_clust, self.side_info.id2rel)
        else:
            self.logger.info('\tLoading cached Clustering')
            self.ent_clust = loadCluster(fname1, self.side_info.ent2id)
            #self.rel_clust = loadCluster(fname2, self.side_info.rel2id)
        '''

    def np_evaluate(self):
        self.logger.info('NP Canonicalizing Evaluation')
        #'''
        cesi_clust2ent = {}
        for rep, cluster in self.ent_clust.items():
            # cesi_clust2ent[rep] = set(cluster)
            cesi_clust2ent[rep] = list(cluster)
        cesi_ent2clust = invertDic(cesi_clust2ent, 'm2os')

        cesi_ent2clust_u = {}
        if self.p.use_assume:
            for trp in self.side_info.triples:
                sub_u, sub = trp['triple_unique'][0], trp['triple'][0]
                cesi_ent2clust_u[sub_u] = cesi_ent2clust[self.side_info.ent2id[sub]]
        else:
            for trp in self.side_info.triples:
                sub_u, sub = trp['triple_unique'][0], trp['triple_unique'][0]
                cesi_ent2clust_u[sub_u] = cesi_ent2clust[self.side_info.ent2id[sub]]

        cesi_clust2ent_u = invertDic(cesi_ent2clust_u, 'm2os')

        #'''
        '''
        cesi_ent2clust_u = {}
        for trp in self.side_info.triples:
            sub_u, sub = trp['triple_unique'][0], trp['triple'][0]
            cesi_ent2clust_u[sub_u] = {self.side_info.ent2clust[self.side_info.ent2id[sub]]}
        f = open('12_cesi_ent2clust_u.csv', 'w')
        for key, value in cesi_ent2clust_u.items():
            f.write(str(key))
            f.write('\t')
            f.write(str(value))
            f.write('\n')
        f.close()
        cesi_clust2ent_u = invertDic(cesi_ent2clust_u, 'm2os')
        '''
        '''
        wrong_cluster, wrong_cluster_should_not, wrong_cluster_should = dict(), dict(), dict()
        should_not_all, should_not_true, should_not_wrong, should_not_wrong_same, should_not_wrong_not_same = 0, 0, 0, 0, 0
        should_not_true_same, should_not_true_not_same = 0, 0
        for key in cesi_clust2ent_u.keys():
            eles = cesi_clust2ent_u[key]
            iter = list(itertools.combinations(eles, 2))
            if len(iter) > 0:
                for i in range(len(iter)):
                    pair = iter[i]
                    cluster1, cluster2 = self.true_ent2clust[pair[0]], self.true_ent2clust[pair[1]]
                    ele1, ele2 = pair[0].split('|')[0], pair[1].split('|')[0]
                    should_not_all += 1
                    if cluster1 == cluster2:
                        should_not_true += 1
                        if ele1 == ele2:
                            should_not_true_same += 1
                        else:
                            should_not_true_not_same += 1
                        continue
                    else:
                        should_not_wrong += 1
                        if pair in wrong_cluster_should_not.keys(): continue
                        else:
                            ele_list = []
                            ele_list.append(key), ele_list.append(cluster1), ele_list.append(cluster2)
                            if ele1 == ele2:
                                should_not_wrong_same += 1
                            else:
                                should_not_wrong_not_same += 1
                            dict_embed1 = self.ent2dict_embed[self.side_info.ent2id[ele1]]
                            dict_embed2 = self.ent2dict_embed[self.side_info.ent2id[ele2]]
                            cos_dict_embed = cos_sim(dict_embed1, dict_embed2)
                            ele_list.append(cos_dict_embed)
                            embed1 = self.ent2embed[self.side_info.ent2id[ele1]]
                            embed2 = self.ent2embed[self.side_info.ent2id[ele2]]
                            cos_embed = cos_sim(embed1, embed2)
                            ele_list.append(cos_embed)
                            wrong_cluster_should_not.update({pair: ele_list})
                            wrong_cluster.update({pair: ele_list})
        f = open('wrong_cluster_should_not.csv', 'w')
        for key, value in wrong_cluster_should_not.items():
            f.write(str(key))
            f.write('\t')
            f.write(str(value))
            f.write('\n')
        f.close()
        print('wrong_cluster_should_not is ok')
        print('should_not_all:', should_not_all)
        print('should_not_true:', should_not_true)
        print('should_not_true_same:', should_not_true_same)
        print('should_not_true_not_same:', should_not_true_not_same)
        print('should_not_wrong:', should_not_wrong)
        print('should_not_wrong_same:', should_not_wrong_same)
        print('should_not_wrong_not_same:', should_not_wrong_not_same)

        should_all, should_true, should_wrong, should_wrong_same, should_wrong_not_same = 0, 0, 0, 0, 0
        should_true_same, should_true_not_same = 0, 0
        for key in self.true_clust2ent.keys():
            eles = self.true_clust2ent[key]
            iter = list(itertools.combinations(eles, 2))
            if len(iter) > 0:
                for i in range(len(iter)):
                    pair = iter[i]
                    cluster1, cluster2 = cesi_ent2clust_u[pair[0]], cesi_ent2clust_u[pair[1]]
                    ele1, ele2 = pair[0].split('|')[0], pair[1].split('|')[0]
                    should_all += 1
                    if cluster1 == cluster2:
                        should_true += 1
                        if ele1 == ele2:
                            should_true_same += 1
                        else:
                            should_true_not_same += 1
                    else:
                        should_wrong += 1
                        if pair in wrong_cluster_should.keys():continue
                        else:
                            ele_list = []
                            ele_list.append(key), ele_list.append(cluster1), ele_list.append(cluster2)
                            if ele1 == ele2:
                                should_wrong_same += 1
                            else:
                                should_wrong_not_same += 1
                            dict_embed1 = self.ent2dict_embed[self.side_info.ent2id[ele1]]
                            dict_embed2 = self.ent2dict_embed[self.side_info.ent2id[ele2]]
                            cos_dict_embed = cos_sim(dict_embed1, dict_embed2)
                            ele_list.append(cos_dict_embed)
                            embed1 = self.ent2embed[self.side_info.ent2id[ele1]]
                            embed2 = self.ent2embed[self.side_info.ent2id[ele2]]
                            cos_embed = cos_sim(embed1, embed2)
                            ele_list.append(cos_embed)
                            wrong_cluster_should.update({pair: ele_list})
                            wrong_cluster.update({pair: ele_list})
        f = open('wrong_cluster_should.csv', 'w')
        for key, value in wrong_cluster_should.items():
            f.write(str(key))
            f.write('\t')
            f.write(str(value))
            f.write('\n')
        f.close()
        f = open('wrong_cluster.csv', 'w')
        for key, value in wrong_cluster.items():
            f.write(str(key))
            f.write('\t')
            f.write(str(value))
            f.write('\n')
        f.close()
        print('wrong_cluster_should is ok')
        print('should_all:', should_all)
        print('should_true:', should_true)
        print('should_true_same:', should_true_same)
        print('should_true_not_same:', should_true_not_same)
        print('should_wrong:', should_wrong)
        print('should_wrong_same:', should_wrong_same)
        print('should_wrong_not_same:', should_wrong_not_same)
        #'''

        eval_results = evaluate(cesi_ent2clust_u, cesi_clust2ent_u, self.true_ent2clust, self.true_clust2ent)

        self.logger.info(
            'Macro prec: {}, Micro prec: {}, Pairwise prec: {}'.format(eval_results['macro_prec'],
                                                                       eval_results['micro_prec'],
                                                                       eval_results['pair_prec']))
        self.logger.info(
            'Macro recall: {}, Micro recall: {}, Pairwise recall: {}'.format(eval_results['macro_recall'],
                                                                       eval_results['micro_recall'],
                                                                       eval_results['pair_recall']))
        self.logger.info(
            'Macro F1: {}, Micro F1: {}, Pairwise F1: {}'.format(eval_results['macro_f1'], eval_results['micro_f1'],
                                                                 eval_results['pair_f1']))
        self.logger.info('CESI: #Clusters: %d, #Singletons %d' % (
        len(cesi_clust2ent_u), len([1 for _, clust in cesi_clust2ent_u.items() if len(clust) == 1])))
        self.logger.info('Gold: #Clusters: %d, #Singletons %d \n' % (
        len(self.true_clust2ent), len([1 for _, clust in self.true_clust2ent.items() if len(clust) == 1])))
        ave_f1 = (eval_results['macro_f1'] + eval_results['micro_f1'] + eval_results['pair_f1']) / 3
        self.logger.info('ave F1: {}'.format(ave_f1))

        # Dump the final results
        fname = self.p.out_path + self.p.file_results
        with open(fname, 'w') as f:
            f.write(json.dumps(eval_results))
        return ave_f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CESI: Canonicalizing Open Knowledge Bases using Embeddings and Side Information')
    # parser.add_argument('-data', dest='dataset', default='reverb45k', help='Dataset to run CESI on:base,ambiguous,reverb45k')
    # parser.add_argument('-split', dest='split', default='test_new', help='Dataset split for evaluation')
    parser.add_argument('-data', dest='dataset', default='OPIEC59k', help='Dataset to run CESI on')
    # parser.add_argument('-split', dest='split', default='valid', help='Dataset split for evaluation')
    parser.add_argument('-split', dest='split', default='test', help='Dataset split for evaluation')
    # parser.add_argument('-data', dest='dataset', default='reverb45k_change', help='Dataset to run CESI on:base,ambiguous,reverb45k')
    # parser.add_argument('-split', dest='split', default='test_read', help='Dataset split for evaluation')
    parser.add_argument('-data_dir', dest='data_dir', default='../data', help='Data directory')
    parser.add_argument('-out_dir', dest='out_dir', default='../output', help='Directory to store CESI output')
    parser.add_argument('-config_dir', dest='config_dir', default='../config', help='Config directory')
    parser.add_argument('-log_dir', dest='log_dir', default='../log', help='Directory for dumping log files')
    parser.add_argument('-ppdb_url', dest='ppdb_url', default='http://localhost:9997/',
                        help='Address of PPDB server')
    # parser.add_argument('-reset', dest="reset", action='store_true',
    #                     help='Clear the cached files (Start a fresh run)')
    parser.add_argument('-reset', dest="reset", action='store_true', default=True,
                        help='Clear the cached files (Start a fresh run)')
    parser.add_argument('-name', dest='name', default=None, help='Assign a name to the run')
    parser.add_argument('-word2vec_path', dest='word2vec_path', default='../init_dict/crawl-300d-2M.vec', help='word2vec_path')
    parser.add_argument('-alignment_module', dest='alignment_module', default='swapping', help='alignment_module')
    parser.add_argument('-Entity_linking_dict_loc', dest='Entity_linking_dict_loc',
                        default='../init_dict/Entity_linking_dict/Whole_Ranked_Merged_Current_dictionary_UTF-8.txt',
                        help='Location of Entity_linking_dict to be loaded')
    parser.add_argument('-change_EL_threshold', dest='change_EL_threshold', default=False, help='change_EL_threshold')
    parser.add_argument('-entity_EL_threshold', dest='entity_EL_threshold', default=0, help='entity_EL_threshold')
    parser.add_argument('-relation_EL_threshold', dest='relation_EL_threshold', default=0, help='relation_EL_threshold')

    # system settings
    parser.add_argument('-embed_init', dest='embed_init', default='crawl', choices=['crawl', 'random'],
                        help='Method for Initializing NP and Relation embeddings')
    parser.add_argument('-embed_loc', dest='embed_loc', default='../init_dict/crawl-300d-2M.vec',
                        help='Location of embeddings to be loaded')

    parser.add_argument('--use_assume', default=True)
    parser.add_argument('--use_Entity_linking_dict', default=True)
    parser.add_argument('--input', default='entity', choices=['entity', 'relation'])

    parser.add_argument('--use_Embedding_model', default=True)
    parser.add_argument('--relation_view_seed_is_web', default=True)
    parser.add_argument('--view_version', default=1.2)  # bert_max_len = 256, ave_len=25, epoch=100 context_view_seed_is_all = True choose_longest_first_sentence = True
    # parser.add_argument('--view_version', default=1.25)  # bert_max_len = 256, ave_len=25, epoch=100 context_view_seed_is_all = False choose_longest_first_sentence = True

    parser.add_argument('--use_cluster_learning', default=False)
    parser.add_argument('--use_cross_seed', default=True)
    parser.add_argument('--combine_seed_and_train_data', default=False)
    # parser.add_argument('--use_cross_seed', default=False)
    # parser.add_argument('--combine_seed_and_train_data', default=True)

    # '--use_cross_seed', default=True '--combine_seed_and_train_data', default=False
    # default=False --single_negative_sample_size = --cross_negative_sample_size = 32 --single_gamma = --cross_gamma = 12.0 --single_batch_size = --cross_batch_size = 2048 50000-1000-1000
    # best_embed_cluster_threshold: 0.52 best_embed_ave_f1: 0.7371666666666666
    # only use KG embedding ...
    # Ave-prec= 0.7759999999999999 macro_prec= 0.7504 micro_prec= 0.8522 pair_prec= 0.7254
    # Ave-recall= 0.7484999999999999 macro_recall= 0.3816 micro_recall= 0.926 pair_recall= 0.9379
    # Ave-F1= 0.7371666666666666 macro_f1= 0.5059 micro_f1= 0.8875 pair_f1= 0.8181
    # Model: #Clusters: 657, #Singletons 7
    # Gold: #Clusters: 490, #Singletons 0

    # best_embed_cluster_threshold: 0.51 best_embed_ave_f1: 0.7335666666666666
    # only use KG embedding ...
    # Ave-prec= 0.7773 macro_prec= 0.7571 micro_prec= 0.8532 pair_prec= 0.7216
    # Ave-recall= 0.7448 macro_recall= 0.3694 micro_recall= 0.9269 pair_recall= 0.9381
    # Ave-F1= 0.7335666666666666 macro_f1= 0.4965 micro_f1= 0.8885 pair_f1= 0.8157
    # Model: #Clusters: 671, #Singletons 7
    # Gold: #Clusters: 490, #Singletons 0

    # '--use_cross_seed', default=False '--combine_seed_and_train_data', default=True 50000
    # best_embed_cluster_threshold: 0.47 best_embed_ave_f1: 0.7257666666666666
    # only use KG embedding ...
    # Ave-prec= 0.7719333333333332 macro_prec= 0.7691 micro_prec= 0.8441 pair_prec= 0.7026
    # Ave-recall= 0.7420666666666667 macro_recall= 0.3592 micro_recall= 0.9244 pair_recall= 0.9426
    # Ave-F1= 0.7257666666666666 macro_f1= 0.4897 micro_f1= 0.8825 pair_f1= 0.8051
    # Model: #Clusters: 693, #Singletons 7
    # Gold: #Clusters: 490, #Singletons 0

    # '--use_cross_seed', default=False '--combine_seed_and_train_data', default=True 100000
    # best_embed_cluster_threshold: 0.54 best_embed_ave_f1: 0.7254999999999999
    # only use KG embedding ...
    # Ave-prec= 0.7342000000000001 macro_prec= 0.7184 micro_prec= 0.8185 pair_prec= 0.6657
    # Ave-recall= 0.7654333333333333 macro_recall= 0.4061 micro_recall= 0.9357 pair_recall= 0.9545
    # Ave-F1= 0.7254999999999999 macro_f1= 0.5189 micro_f1= 0.8732 pair_f1= 0.7844
    # Model: #Clusters: 618, #Singletons 5
    # Gold: #Clusters: 490, #Singletons 0

    parser.add_argument('--use_soft_learning', default=False)
    parser.add_argument('--update_seed', default=False)
    parser.add_argument('--only_update_sim', default=True)

    parser.add_argument('--use_bert_update_seeds', default=False)
    parser.add_argument('--use_new_embedding', default=False)
    # crawl + TransE + new seed + update seed

    parser.add_argument('--max_steps', default=50000, type=int)
    parser.add_argument('--turn_to_seed', default=1000, type=int)
    # parser.add_argument('--seed_max_steps', default=2000, type=int)
    parser.add_argument('--seed_max_steps', default=1000, type=int)
    parser.add_argument('--update_seed_steps', default=6000, type=int)

    parser.add_argument('--get_new_cross_seed', default=True)
    parser.add_argument('--entity_threshold', dest='entity_threshold', default=0.9, help='entity_threshold')
    parser.add_argument('--relation_threshold', dest='relation_threshold', default=0.95, help='relation_threshold')

    parser.add_argument('--use_context', default=True)
    parser.add_argument('--use_attention', default=True)
    parser.add_argument('--replace_h', default=True)
    parser.add_argument('--sentence_delete_stopwords', default=True)
    parser.add_argument('--use_first_sentence', default=True)
    parser.add_argument('--use_BERT', default=True)

    # Multi-view
    parser.add_argument('--step_0_use_hac', default=False)

    # HAN
    #parser.add_argument('--data_path', type=str, default='./sample_text.csv')
    #parser.add_argument('--min_word_count', type=int, default=5)

    # parser.add_argument('--epochs', type=int, default=300)
    # parser.add_argument('--batch_size', type=int, default=50)
    # parser.add_argument("--device", default="/gpu:0")
    # parser.add_argument("--lr", type=float, default=0.001)

    # RotatE
    parser.add_argument('--cuda', action='store_true', help='use GPU', default=True)
    parser.add_argument('--do_train', action='store_true', default=True)
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data', default=False)
    parser.add_argument('--save_path', default='../output', type=str)

    # parser.add_argument('--model', default='new_rotate', type=str)
    parser.add_argument('--model', default='TransE', type=str)
    parser.add_argument('-de', '--double_entity_embedding', action='store_true', default=False)
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true', default=False)

    parser.add_argument('-d', '--hidden_dim', default=300, type=int)
    parser.add_argument('-n1', '--single_negative_sample_size', default=32, type=int)
    # parser.add_argument('-n2', '--cross_negative_sample_size', default=40, type=int)
    parser.add_argument('-n2', '--cross_negative_sample_size', default=32, type=int)
    parser.add_argument('-g1', '--single_gamma', default=12.0, type=float)
    parser.add_argument('-g2', '--cross_gamma', default=12.0, type=float)
    # parser.add_argument('-g2', '--cross_gamma', default=0.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true', default=True)
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b1', '--single_batch_size', default=2048, type=int)
    parser.add_argument('-b2', '--cross_batch_size', default=2048, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true',
                        help='Otherwise use subsampling weighting like in word2vec', default=True)

    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=12, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('--warm_up_steps', default=None, type=int)

    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')

    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('-embed_dims', dest='embed_dims', default=300, type=int, help='Embedding dimension')

    # word2vec and iteration hyper-parameters
    parser.add_argument('-retrain_literal_embeds', dest='retrain_literal_embeds', default=True,
                        help='retrain_literal_embeds')

    # Clustering hyper-parameters
    parser.add_argument('-linkage', dest='linkage', default='complete', choices=['complete', 'single', 'average'],
                        help='HAC linkage criterion')
    # parser.add_argument('-thresh_val', dest='thresh_val', default=.4239, type=float, help='Threshold for clustering')
    # parser.add_argument('-thresh_val', dest='thresh_val', default=cluster_threshold_real, type=float,
                        #help='Threshold for clustering')
    parser.add_argument('-metric', dest='metric', default='cosine',
                        help='Metric for calculating distance between embeddings')
    parser.add_argument('-num_canopy', dest='num_canopy', default=1, type=int,
                        help='Number of caponies while clustering')
    parser.add_argument('-true_seed_num', dest='true_seed_num', default=2361, type=int)
    args = parser.parse_args()

    # if args.name == None: args.name = args.dataset + '_' + args.split + '_' + time.strftime(
    #     "%d_%m_%Y") + '_' + time.strftime("%H:%M:%S")
    if args.name == None: args.name = args.dataset + '_' + args.split + '_' + '1'

    args.file_triples = '/triples.txt'  # Location for caching triples
    args.file_entEmbed = '/embed_ent.pkl'  # Location for caching learned embeddings for noun phrases
    args.file_relEmbed = '/embed_rel.pkl'  # Location for caching learned embeddings for relation phrases
    args.file_entClust = '/cluster_ent.txt'  # Location for caching Entity clustering results
    args.file_relClust = '/cluster_rel.txt'  # Location for caching Relation clustering results
    args.file_sideinfo = '/side_info.txt'  # Location for caching side information extracted for the KG (for display)
    args.file_sideinfo_pkl = '/side_info.pkl'  # Location for caching side information extracted for the KG (binary)
    args.file_hyperparams = '/hyperparams.json'  # Location for loading hyperparameters
    args.file_results = '/results.json'  # Location for loading hyperparameters

    args.out_path = args.out_dir + '/' + args.name  # Directory for storing output
    print('args.log_dir:', args.log_dir)
    print('args.out_path:', args.out_path)
    print('args.reset:', args.reset)
    args.data_path = args.data_dir + '/' + args.dataset + '/' + args.dataset + '_' + args.split  # Path to the dataset
    if args.reset: os.system('rm -r {}'.format(args.out_path))  # Clear cached files if requeste
    if not os.path.isdir(args.out_path): os.system(
        'mkdir -p ' + args.out_path)  # Create the output directory if doesn't exist

    cesi = CESI_Main(args)  # Loading KG triples
    cesi.get_sideInfo()  # Side Information Acquisition
    cesi.embedKG()  # Learning embedding for Noun and relation phrases
    # cesi.HAN()
    #cesi.cluster()  # Clustering NP and relation phrase embeddings
    #cesi.np_evaluate()  # Evaluating the performance over NP canonicalization

    best_threshold = dict()
    best_cluster_threshold, best_ave_f1 = 0, 0
    #'''
    cluster_threshold_max = 50
    cluster_threshold_min = 10
    for cluster_threshold in range(cluster_threshold_max, cluster_threshold_min, -1):
        cluster_threshold_real = cluster_threshold / 100
    #for cluster_threshold in range(140, -10, -10):
        #cluster_threshold_real = cluster_threshold / 1000
        print('cluster_threshold_real:', cluster_threshold_real)
        cesi.cluster(cluster_threshold_real)  # Clustering NP and relation phrase embeddings
        ave_f1 = cesi.np_evaluate()  # Evaluating the performance over NP canonicalization
        best_threshold.update({cluster_threshold_real: ave_f1})
    for cluster_threshold in range(cluster_threshold_max, cluster_threshold_min, -1):
        cluster_threshold_real = cluster_threshold / 100
        value = best_threshold[cluster_threshold_real]
        if value > best_ave_f1:
            best_cluster_threshold = cluster_threshold_real
            best_ave_f1 = value
        else:
            continue
    print(best_threshold)
    print('best_cluster_threshold:', best_cluster_threshold, 'best_ave_f1:', best_ave_f1)