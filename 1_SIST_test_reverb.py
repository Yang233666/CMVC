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

        self.id2ent_unique, self.id2rel_unique, self.id2ent_true = dict(), dict(), dict()

        print('dataset:', args.dataset)
        num = 0
        if args.dataset == 'OPIEC':
            print('load OPIEC_dataset ... ')
            self.triples_list = pickle.load(open(args.data_path, 'rb'))

            ''' Ground truth clustering '''
            self.true_ent2clust = ddict(set)
            for trp in self.triples_list:

                sub_u = trp['triple_unique'][0]
                self.true_ent2clust[sub_u].add(trp['subject_wiki_link'])

                self.triples_list.append(trp)

                self.id2ent_unique[str(num)] = trp['triple_unique'][0]
                self.id2rel_unique[str(num)] = trp['triple_unique'][1]
                self.id2ent_true[str(num)] = trp['subject_wiki_link']

                num += 1
                if num == 3000:
                    break

            self.true_clust2ent = invertDic(self.true_ent2clust, 'm2os')

            # SIST_result_file = '../file/ICDE19_result/' + self.p.dataset + '/OPIEC_test_0.1.json'
            # SIST_result_file = '../file/ICDE19_result/' + self.p.dataset + '/OPIEC_test_0.2.json'
            # SIST_result_file = '../file/ICDE19_result/' + self.p.dataset + '/OPIEC_test_0.3.json'
            # SIST_result_file = '../file/ICDE19_result/' + self.p.dataset + '/OPIEC_test_0.4.json'
            SIST_result_file = '../file/ICDE19_result/' + self.p.dataset + '/OPIEC_test_0.5.json'

        else:
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
                    self.id2ent_unique[str(trp['_id'])] = trp['triple_unique'][0]
                    self.id2rel_unique[str(trp['_id'])] = rel + '|' + str(trp['_id'])
                    self.id2ent_true[str(trp['_id'])] = trp['true_sub_link']

                    # num += 1
                    # if num == 3000:
                    #     break

                    # SIST_result_file = '../file/ICDE19_result/' + self.p.dataset + '/reverb45k_test_0.1.json'
                    # SIST_result_file = '../file/ICDE19_result/' + self.p.dataset + '/reverb45k_test_0.2.json'
                    # SIST_result_file = '../file/ICDE19_result/' + self.p.dataset + '/reverb45k_test_0.3.json'
                    # SIST_result_file = '../file/ICDE19_result/' + self.p.dataset + '/reverb45k_test_0.4.json'
                    SIST_result_file = '../file/ICDE19_result/' + self.p.dataset + '/reverb45k_test_0.5.json'

        print('self.triples_list:', type(self.triples_list), len(self.triples_list))
        print('self.id2ent_unique:', type(self.id2ent_unique), len(self.id2ent_unique))
        print('self.id2rel_unique', type(self.id2rel_unique) ,len(self.id2rel_unique))
        print('self.id2ent_true:', type(self.id2ent_true), len(self.id2ent_true))
        print('SIST_result_file:', SIST_result_file)

        file = open(SIST_result_file, 'r', encoding='utf-8')
        SIST_result = []
        for line in file.readlines():
            if ' ' in line:
                cluster = line.split()
            else:
                cluster = [line.replace('\n', '')]
            if cluster not in SIST_result:
                SIST_result.append(cluster)
        print('SIST_result:', type(SIST_result), len(SIST_result), SIST_result)

        cesi_clust2ent_u = {}
        # self.true_ent2clust = ddict(set)
        for i in range(len(SIST_result)):
            id_list = SIST_result[i]
            clust_list = []
            for id in id_list:
                if id in self.id2ent_unique:
                    sub_u = self.id2ent_unique[id]
                    clust_list.append(sub_u)
                    # self.true_ent2clust[sub_u].add(self.id2ent_true[id])
            cesi_clust2ent_u[i] = set(clust_list)

        cesi_ent2clust_u = invertDic(cesi_clust2ent_u, 'm2os')
        # self.true_clust2ent = invertDic(self.true_ent2clust, 'm2os')

        ''' Ground truth clustering '''
        self.true_ent2clust = ddict(set)
        for trp in self.triples_list:
            sub_u = trp['triple_unique'][0]
            self.true_ent2clust[sub_u].add(trp['true_sub_link'])
            # obj_u = trp['triple_unique'][2]
            # self.true_ent2clust[obj_u].add(trp['true_obj_link'])
        self.true_clust2ent = invertDic(self.true_ent2clust, 'm2os')

        print('cesi_clust2ent_u:', type(cesi_clust2ent_u), len(cesi_clust2ent_u))
        print('cesi_ent2clust_u:', type(cesi_ent2clust_u), len(cesi_ent2clust_u))
        print('self.true_clust2ent:', type(self.true_clust2ent), len(self.true_clust2ent))
        print('self.true_ent2clust:', type(self.true_ent2clust), len(self.true_ent2clust))
        # cesi_clust2ent_u: <class 'dict'> 168
        # cesi_ent2clust_u: <class 'dict'> 35
        # self.true_clust2ent: <class 'dict'> 4
        # self.true_ent2clust: <class 'collections.defaultdict'> 35

        eval_results = evaluate(cesi_ent2clust_u, cesi_clust2ent_u, self.true_ent2clust, self.true_clust2ent)
        macro_prec, micro_prec, pair_prec = eval_results['macro_prec'], eval_results['micro_prec'], eval_results[
            'pair_prec']
        macro_recall, micro_recall, pair_recall = eval_results['macro_recall'], eval_results['micro_recall'], \
                                                  eval_results['pair_recall']
        macro_f1, micro_f1, pair_f1 = eval_results['macro_f1'], eval_results['micro_f1'], eval_results['pair_f1']
        ave_prec = (macro_prec + micro_prec + pair_prec) / 3
        ave_recall = (macro_recall + micro_recall + pair_recall) / 3
        ave_f1 = (macro_f1 + micro_f1 + pair_f1) / 3
        model_clusters = len(cesi_clust2ent_u)
        model_Singletons = len([1 for _, clust in cesi_clust2ent_u.items() if len(clust) == 1])
        gold_clusters = len(self.true_clust2ent)
        gold_Singletons = len([1 for _, clust in self.true_clust2ent.items() if len(clust) == 1])
        print('Ave-prec=', ave_prec, 'macro_prec=', macro_prec, 'micro_prec=', micro_prec,
              'pair_prec=', pair_prec)
        print('Ave-recall=', ave_recall, 'macro_recall=', macro_recall, 'micro_recall=', micro_recall,
              'pair_recall=', pair_recall)
        print('Ave-F1=', ave_f1, 'macro_f1=', macro_f1, 'micro_f1=', micro_f1, 'pair_f1=', pair_f1)
        print('Model: #Clusters: %d, #Singletons %d' % (model_clusters, model_Singletons))
        print('Gold: #Clusters: %d, #Singletons %d' % (gold_clusters, gold_Singletons))
        print()
        exit()





if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CESI: Canonicalizing Open Knowledge Bases using Embeddings and Side Information')
    # parser.add_argument('-data', dest='dataset', default='reverb45k', help='Dataset to run CESI on:base,ambiguous,reverb45k')
    # parser.add_argument('-split', dest='split', default='test_new', help='Dataset split for evaluation')
    # parser.add_argument('-split', dest='split', default='test', help='Dataset split for evaluation')
    # parser.add_argument('-data', dest='dataset', default='OPIEC', help='Dataset to run CESI on')
    # parser.add_argument('-split', dest='split', default='53k', help='Dataset split for evaluation')
    parser.add_argument('-data', dest='dataset', default='reverb45k_change', help='Dataset to run CESI on:base,ambiguous,reverb45k')
    parser.add_argument('-split', dest='split', default='test_read', help='Dataset split for evaluation')
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
    parser.add_argument('--use_soft_learning', default=False)

    parser.add_argument('--update_seed', default=False)
    parser.add_argument('--only_update_sim', default=True)

    parser.add_argument('--use_bert_update_seeds', default=False)
    parser.add_argument('--use_new_embedding', default=False)
    # crawl + TransE + new seed + update seed

    parser.add_argument('--max_steps', default=50000, type=int)
    parser.add_argument('--turn_to_seed', default=1000, type=int)
    parser.add_argument('--seed_max_steps', default=2000, type=int)
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

    parser.add_argument('-n1', '--single_negative_sample_size', default=32, type=int)
    # parser.add_argument('-n1', '--single_negative_sample_size', default=2, type=int)
    parser.add_argument('-n2', '--cross_negative_sample_size', default=40, type=int)
    parser.add_argument('-d', '--hidden_dim', default=300, type=int)
    parser.add_argument('-g1', '--single_gamma', default=12.0, type=float)
    parser.add_argument('-g2', '--cross_gamma', default=0.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true', default=True)
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b1', '--single_batch_size', default=2048, type=int)
    # parser.add_argument('-b1', '--single_batch_size', default=48, type=int)
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
    # cesi.get_sideInfo()  # Side Information Acquisition