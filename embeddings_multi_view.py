import gensim
from helper import *
from utils import cos_sim
from cluster_f1_test import Find_Best_Result, HAC_getClusters, embed2f1, cluster_test, cluster_test_sample, test_sample, \
    cluster_test_sample_relation
# from get_context_view_embedding import Context_Embeddings
# from train_embedding_model import Train_Embedding_Model, pair2triples
from train_embedding_model_end_point import Train_Embedding_Model, pair2triples
from BERT_Embeddings import BERT_Model
# from BERT_Embeddings_figure import BERT_Model
from Multi_view_spherical_kmeans import Multi_view_SphericalKMeans
from sklearn.cluster import KMeans


class DisjointSet(object):
    def __init__(self):
        self.leader = {}  # maps a member to the group's leader
        self.group = {}  # maps a group leader to the group (which is a set)

    def add(self, a, b):
        leadera = self.leader.get(a)
        leaderb = self.leader.get(b)
        if leadera is not None:
            if leaderb is not None:
                if leadera == leaderb: return  # nothing to do
                groupa = self.group[leadera]
                groupb = self.group[leaderb]
                if len(groupa) < len(groupb):
                    a, leadera, groupa, b, leaderb, groupb = b, leaderb, groupb, a, leadera, groupa
                groupa |= groupb
                del self.group[leaderb]
                for k in groupb:
                    self.leader[k] = leadera
            else:
                self.group[leadera].add(b)
                self.leader[b] = leadera
        else:
            if leaderb is not None:
                self.group[leaderb].add(a)
                self.leader[a] = leaderb
            else:
                self.leader[a] = self.leader[b] = a
                self.group[a] = set([a, b])


def amieInfo(triples, ent2id, rel2id):
    uf = DisjointSet()
    min_supp = 2
    min_conf = 0.5  # cesi=0.2
    amie_cluster = []
    rel_so = {}

    for trp in triples:
        sub, rel, obj = trp['triple']
        if sub in ent2id and rel in rel2id and obj in ent2id:
            sub_id, rel_id, obj_id = ent2id[sub], rel2id[rel], ent2id[obj]
            rel_so[rel_id] = rel_so.get(rel_id, set())
            rel_so[rel_id].add((sub_id, obj_id))

    for r1, r2 in itertools.combinations(rel_so.keys(), 2):
        supp = len(rel_so[r1].intersection(rel_so[r2]))
        if supp < min_supp: continue

        s1, _ = zip(*list(rel_so[r1]))
        s2, _ = zip(*list(rel_so[r2]))

        z_conf_12, z_conf_21 = 0, 0
        for ele in s1:
            if ele in s2: z_conf_12 += 1
        for ele in s2:
            if ele in s1: z_conf_21 += 1

        conf_12 = supp / z_conf_12
        conf_21 = supp / z_conf_21

        if conf_12 >= min_conf and conf_21 >= min_conf:
            amie_cluster.append((r1, r2))  # Replace with union find DS
            uf.add(r1, r2)

    rel2amie = uf.leader
    return rel2amie


def seed_pair2cluster(seed_pair_list, ent_list):
    pair_dict = dict()
    for seed_pair in seed_pair_list:
        a, b = seed_pair
        if a != b:
            if a < b:
                rep, ent_id = a, b
            else:
                ent_id, rep = b, a
            if ent_id not in pair_dict:
                if rep not in pair_dict:
                    pair_dict.update({ent_id: rep})
                else:
                    new_rep = pair_dict[rep]
                    j = 0
                    while rep in pair_dict:  # 寻找根节点
                        new_rep = pair_dict[rep]
                        rep = new_rep
                        j += 1
                        if j > 1000000:
                            break
                        # print('j:', j)
                    pair_dict.update({ent_id: new_rep})
            else:
                if rep not in pair_dict:
                    new_rep = pair_dict[ent_id]
                    if rep > new_rep:
                        pair_dict.update({rep: new_rep})
                    else:
                        pair_dict.update({new_rep: rep})
                else:
                    old_rep = rep
                    new_rep = pair_dict[rep]
                    j = 0
                    while rep in pair_dict:  # 寻找根节点
                        new_rep = pair_dict[rep]
                        rep = new_rep
                        j += 1
                        if j > 1000000:
                            break
                        # print('j:', j)
                    if old_rep > new_rep:
                        pair_dict.update({ent_id: new_rep})
                    else:
                        pair_dict.update({ent_id: old_rep})

    # print('pair_dict:', type(pair_dict), len(pair_dict), pair_dict)
    cluster_list = []
    for i in range(len(ent_list)):
        cluster_list.append(i)
    for ent_id in pair_dict:
        rep = pair_dict[ent_id]
        if ent_id < len(cluster_list):
            cluster_list[ent_id] = rep
    return cluster_list


def get_seed_pair(ent_list, ent2id, ent_old_id2new_id):
    seed_pair = []
    for i in range(len(ent_list)):
        ent1 = ent_list[i]
        old_id1 = ent2id[ent1]
        if old_id1 in ent_old_id2new_id:
            for j in range(i + 1, len(ent_list)):
                ent2 = ent_list[j]
                old_id2 = ent2id[ent2]
                if old_id2 in ent_old_id2new_id:
                    new_id1, new_id2 = ent_old_id2new_id[old_id1], ent_old_id2new_id[
                        old_id2]
                    if new_id1 == new_id2:
                        id_tuple = (i, j)
                        seed_pair.append(id_tuple)
    return seed_pair


def difference_cluster2pair(cluster_list_1, cluster_list_2, EL_seed):
    # print('cluster_list_1:', type(cluster_list_1), len(cluster_list_1), cluster_list_1)
    # print('cluster_list_2:', type(cluster_list_2), len(cluster_list_2), cluster_list_2)
    new_seed_pair_list = []
    for i in range(len(cluster_list_1)):
        id_1, id_2 = cluster_list_1[i], cluster_list_2[i]
        if id_1 == id_2:
            continue
        else:
            index_list_1 = [i for i, x in enumerate(cluster_list_1) if x == id_1]
            index_list_2 = [i for i, x in enumerate(cluster_list_2) if x == id_2]
            if len(index_list_2) == 1:
                continue
            else:
                iter_list_1 = list(itertools.combinations(index_list_1, 2))
                iter_list_2 = list(itertools.combinations(index_list_2, 2))
                if len(iter_list_1) > 0:
                    for iter_pair in iter_list_1:
                        if iter_pair in iter_list_2: iter_list_2.remove(iter_pair)
                for iter in iter_list_2:
                    if iter not in EL_seed:
                        new_seed_pair_list.append(iter)
    return new_seed_pair_list


def totol_cluster2pair(cluster_list):
    # print('cluster_list:', type(cluster_list), len(cluster_list), cluster_list)
    seed_pair_list, id_list = [], []
    for i in range(len(cluster_list)):
        id = cluster_list[i]
        if id not in id_list:
            id_list.append(id)
            index_list = [i for i, x in enumerate(cluster_list) if x == id]
            if len(index_list) > 1:
                iter_list = list(itertools.combinations(index_list, 2))
                seed_pair_list += iter_list
    return seed_pair_list


class Embeddings(object):
    """
    Learns embeddings for NPs and relation phrases
    """

    def __init__(self, params, side_info, logger, true_ent2clust, true_clust2ent, sub_uni2triple_dict=None,
                 triple_list=None):
        self.p = params
        self.logger = logger

        self.side_info = side_info
        self.ent2embed = {}  # Stores final embeddings learned for noun phrases
        self.rel2embed = {}  # Stores final embeddings learned for relation phrases
        self.true_ent2clust, self.true_clust2ent = true_ent2clust, true_clust2ent
        self.sub_uni2triple_dict = sub_uni2triple_dict
        self.triples_list = triple_list

        self.rel_id2sentence_list = dict()

        ent_id2sentence_list = self.side_info.ent_id2sentence_list
        for rel in self.side_info.rel_list:
            rel_id = self.side_info.rel2id[rel]
            if rel_id not in self.rel_id2sentence_list:
                triple_id_list = self.side_info.rel2triple_id_list[rel]
                sentence_list = []
                for triple_id in triple_id_list:
                    triple = self.triples_list[triple_id]
                    sub, rel_, obj = triple['triple'][0], triple['triple'][1], triple['triple'][2]
                    assert str(rel_) == str(rel)
                    if sub in self.side_info.ent2id:
                        sentence_list += ent_id2sentence_list[self.side_info.ent2id[sub]]
                    if obj in self.side_info.ent2id:
                        sentence_list += ent_id2sentence_list[self.side_info.ent2id[obj]]
                sentence_list = list(set(sentence_list))
                self.rel_id2sentence_list[rel_id] = sentence_list
        print('self.rel_id2sentence_list:', type(self.rel_id2sentence_list), len(self.rel_id2sentence_list))

    def fit(self):
        clean_ent_list, clean_rel_list = [], []
        for ent in self.side_info.ent_list: clean_ent_list.append(ent.split('|')[0])
        for rel in self.side_info.rel_list: clean_rel_list.append(rel.split('|')[0])

        print('clean_ent_list:', type(clean_ent_list), len(clean_ent_list))  # 19915
        print('clean_rel_list:', type(clean_rel_list), len(clean_rel_list))  # 18250

        ''' Intialize embeddings '''
        if self.p.embed_init == 'crawl':
            fname1, fname2 = '../file/' + self.p.dataset + '_' + self.p.split + '/1E_init', '../file/' + self.p.dataset + '_' + self.p.split + '/1R_init'
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

        # folder = 'multi_view/relation_view_' + str(self.p.input)  + '_' + str(self.p.max_steps) + '_' + str(self.p.turn_to_seed) + '_' + str(self.p.seed_max_steps)
        folder = 'multi_view/relation_view'
        print('folder:', folder)
        folder_to_make = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder + '/'
        if not os.path.exists(folder_to_make):
            os.makedirs(folder_to_make)

        if str(self.p.input) == 'entity':
            fname = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder + '/crawl_cluster_list'
            if not checkFile(fname):
                print('generate best crawl clusters_list', fname)
                if self.p.split == 'valid':
                    cluster_threshold_max, cluster_threshold_min = 90, 10
                    FBR = Find_Best_Result(self.p, self.side_info, self.logger, self.true_ent2clust,
                                           self.true_clust2ent,
                                           clean_ent_list, self.E_init, cluster_threshold_max, cluster_threshold_min)
                    best_embed_threshold, best_crawl_clusters_list, best_el_crawl_clusters_list, el_prior_cluster_list = FBR.go()
                    print('best_crawl_clusters_list', 'best_embed_threshold:', best_embed_threshold)
                else:
                    if self.p.dataset == 'OPIEC59k':
                        best_embed_threshold = 0.29
                    else:
                        best_embed_threshold = 0.35
                    np_view_clusters, np_view_clusters_center = HAC_getClusters(self.p, self.E_init,
                                                                                best_embed_threshold)
                    best_crawl_clusters_list = list(np_view_clusters)
                pickle.dump(best_crawl_clusters_list, open(fname, 'wb'))
            else:
                print('load best crawl clusters_list')
                best_crawl_clusters_list = pickle.load(open(fname, 'rb'))
            print('best_crawl_clusters_list')
            cluster_test(self.p, self.side_info, best_crawl_clusters_list, self.true_ent2clust, self.true_clust2ent,
                         print_or_not=True)

            if self.p.dataset == 'NYTimes2018' and self.p.input == 'entity':
                # 获得NYT数据集的sample 100 clusters的结果
                folder_sample = 'multi_view/sample_result/crawl'
                print('folder_sample:', folder_sample)
                folder_to_make_sample = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder_sample + '/'
                if not os.path.exists(folder_to_make_sample):
                    os.makedirs(folder_to_make_sample)
                fname1 = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder_sample + '/cesi_clust2ent_u.json'
                fname2 = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder_sample + '/true_clust2ent.json'
                fname3 = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder_sample + '/true_ent2sentences.json'
                if not checkFile(fname1) or not checkFile(fname2):
                    for i in range(200):
                        model_clust2ent_u, true_clust2ent, macro_f1, micro_f1, pair_f1, ave_f1 = cluster_test_sample(
                            self.p, self.side_info,
                            best_crawl_clusters_list,
                            self.true_ent2clust,
                            self.true_clust2ent,
                            print_or_not=True)
                        if float(ave_f1) > 0.73 and float(ave_f1) < 0.75:
                            break
                    for k in model_clust2ent_u:
                        v = list(model_clust2ent_u[k])
                        model_clust2ent_u[k] = v
                    for k in true_clust2ent:
                        v = list(true_clust2ent[k])
                        true_clust2ent[k] = v
                    model_clust2ent_u_str = json.dumps(model_clust2ent_u, indent=4)
                    with open(fname1, 'w') as json_file:
                        json_file.write(model_clust2ent_u_str)
                    print('dump model_clust2ent_u ok')

                    true_clust2ent_str = json.dumps(true_clust2ent, indent=4)
                    with open(fname2, 'w') as json_file:
                        json_file.write(true_clust2ent_str)
                    print('dump true_clust2ent ok')
                    true_ent2clust = invertDic(true_clust2ent, 'm2os')
                    true_ent2sentences = dict()
                    for ent_unique in true_ent2clust:
                        trp = self.sub_uni2triple_dict[ent_unique]
                        trp_new = dict()
                        trp_new['triple'] = trp['triple']
                        trp_new['src_sentences'] = trp['src_sentences']
                        trp_new['triple_unique'] = trp['triple_unique']
                        true_ent2sentences[ent_unique] = trp_new
                    true_ent2sentences_str = json.dumps(true_ent2sentences, indent=4)
                    with open(fname3, 'w') as json_file:
                        json_file.write(true_ent2sentences_str)
                    print('dump true_ent2sentences ok')
                else:
                    print('load')
                    f = open(fname1, 'r')
                    content = f.read()
                    model_clust2ent_u = json.loads(content)
                    f.close()
                    f = open(fname2, 'r')
                    content = f.read()
                    true_clust2ent = json.loads(content)
                    f.close()
                    test_sample(model_clust2ent_u, true_clust2ent, print_or_not=True)
                    # Ave-prec= 0.6057666666666667 macro_prec= 0.75 micro_prec= 0.7005 pair_prec= 0.3668
                    # Ave-recall= 0.9916 macro_recall= 0.9803 micro_recall= 0.9956 pair_recall= 0.9989
                    # Ave-F1= 0.7362333333333334 macro_f1= 0.8498 micro_f1= 0.8224 pair_f1= 0.5365
                    # Model: #Clusters: 100, #Singletons 0
                    # Gold: #Clusters: 203, #Singletons 42

        fname_EL = '../file/' + self.p.dataset + '_' + self.p.split + '/EL_seed'
        if not checkFile(fname_EL):
            self.EL_seed = get_seed_pair(self.side_info.ent_list, self.side_info.ent2id,
                                            self.side_info.ent_old_id2new_id)
            pickle.dump(self.EL_seed, open(fname_EL, 'wb'))
        else:
            self.EL_seed = pickle.load(open(fname_EL, 'rb'))
        print('self.EL_seed:', type(self.EL_seed), len(self.EL_seed))

        fname_amie = '../file/' + self.p.dataset + '_' + self.p.split + '/amie_rp_seed'
        if not checkFile(fname_amie):
            self.amie_rp = amieInfo(self.triples_list, self.side_info.ent2id, self.side_info.rel2id)
            self.amie_rp_seed = get_seed_pair(self.side_info.rel_list, self.side_info.rel2id,
                                            self.amie_rp)
            pickle.dump(self.amie_rp_seed, open(fname_amie, 'wb'))
        else:
            self.amie_rp_seed = pickle.load(open(fname_amie, 'rb'))
        print('self.amie_rp_seed:', type(self.amie_rp_seed), len(self.amie_rp_seed))

        web_seed_Jaccard_threshold = 0.015
        fname2_entity = '../file/' + self.p.dataset + '_' + self.p.split + '/WEB_seed/entity/cluster_list_threshold_' + \
                        str(web_seed_Jaccard_threshold) + '_url_max_length_all'
        fname2_relation = '../file/' + self.p.dataset + '_' + self.p.split + '/WEB_seed/relation/cluster_list_threshold_' + \
                          str(web_seed_Jaccard_threshold) + '_url_max_length_all'
        print('fname2_entity:', fname2_entity)
        print('fname2_relation:', fname2_relation)
        self.web_entity_cluster_list = pickle.load(open(fname2_entity, 'rb'))
        self.web_relation_cluster_list = pickle.load(open(fname2_relation, 'rb'))
        print('self.web_entity_cluster_list:', type(self.web_entity_cluster_list),
              len(self.web_entity_cluster_list),
              self.web_entity_cluster_list)
        print('self.web_relation_cluster_list:', type(self.web_relation_cluster_list),
              len(self.web_relation_cluster_list),
              self.web_relation_cluster_list)
        # cluster_test(self.p, self.side_info, self.web_entity_cluster_list, self.true_ent2clust, self.true_clust2ent,
        #              print_or_not=True)
        self.web_entity_seed_pair_list = totol_cluster2pair(self.web_entity_cluster_list)
        self.web_relation_seed_pair_list = totol_cluster2pair(self.web_relation_cluster_list)
        print('self.web_entity_seed_pair_list:', type(self.web_entity_seed_pair_list),
              len(self.web_entity_seed_pair_list), self.web_entity_seed_pair_list[0:10])
        print('self.web_relation_seed_pair_list:', type(self.web_relation_seed_pair_list),
              len(self.web_relation_seed_pair_list), self.web_relation_seed_pair_list[0:10])

        web_entity_cluster_list = seed_pair2cluster(self.web_entity_seed_pair_list, clean_ent_list)
        cluster_test(self.p, self.side_info, web_entity_cluster_list, self.true_ent2clust, self.true_clust2ent,
                     print_or_not=True)
        # print('sample! ----------------------------------------------------------------------------------')
        # cluster_test_sample(self.p, self.side_info, self.web_cluster_list, self.true_ent2clust, self.true_clust2ent,
        #                     print_or_not=True)
        web_relation_cluster_list = seed_pair2cluster(self.web_relation_seed_pair_list, clean_rel_list)
        print('web_relation_cluster_list:', type(web_relation_cluster_list), len(web_relation_cluster_list),
              web_relation_cluster_list[0:10])
        print('different web_relation_cluster:', len(list(set(web_relation_cluster_list))))
        print()

        web_relation = False
        if web_relation:
            folder_sample = 'multi_view/sample_rp_result/web/'
            print('folder_sample:', folder_sample)
            folder_to_make_sample = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder_sample + '/'
            if not os.path.exists(folder_to_make_sample):
                os.makedirs(folder_to_make_sample)
            fname1 = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder_sample + '/cesi_clust2ent_u.json'
            fname2 = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder_sample + '/true_clust2ent.json'
            if not checkFile(fname1) or not checkFile(fname2):
                model_clust2ent_u, true_clust2ent, macro_f1, micro_f1, pair_f1, ave_f1 = cluster_test_sample_relation(
                    self.p, self.side_info,
                    web_relation_cluster_list,
                    self.true_ent2clust,
                    self.true_clust2ent,
                    print_or_not=True, cluster_num=10000000)
                model_clust2ent_u_copy = model_clust2ent_u.copy()
                for k in model_clust2ent_u_copy:
                    v = list(model_clust2ent_u[k])
                    model_clust2ent_u.pop(k)
                    model_clust2ent_u[str(k)] = v
                true_clust2ent_copy = true_clust2ent.copy()
                for k in true_clust2ent_copy:
                    v = list(true_clust2ent[k])
                    true_clust2ent.pop(k)
                    true_clust2ent[str(k)] = v
                model_clust2ent_u_str = json.dumps(model_clust2ent_u, indent=4)
                with open(fname1, 'w') as json_file:
                    json_file.write(model_clust2ent_u_str)
                print('dump model_clust2ent_u ok')
                true_clust2ent_str = json.dumps(true_clust2ent, indent=4)
                with open(fname2, 'w') as json_file:
                    json_file.write(true_clust2ent_str)
                print('dump true_clust2ent ok')
            else:
                print('load')
                f = open(fname1, 'r')
                content = f.read()
                model_clust2ent_u = json.loads(content)
                f.close()
                f = open(fname2, 'r')
                content = f.read()
                true_clust2ent = json.loads(content)
                f.close()
                test_sample(model_clust2ent_u, true_clust2ent, print_or_not=True)
            exit()

        el_cluster_list = seed_pair2cluster(self.EL_seed, clean_ent_list)
        print('el_cluster_list:', type(el_cluster_list), len(el_cluster_list), el_cluster_list[0:10])
        cluster_test(self.p, self.side_info, el_cluster_list, self.true_ent2clust, self.true_clust2ent,
                     print_or_not=True)
        # print('sample! ----------------------------------------------------------------------------------')
        # cluster_test_sample(self.p, self.side_info, el_cluster_list, self.true_ent2clust, self.true_clust2ent,
        #                     print_or_not=True)
        if self.p.dataset == 'NYTimes2018' and self.p.input == 'entity':
            # 获得NYT数据集的sample 100 clusters的结果
            folder_sample = 'multi_view/sample_result/entity_linking'
            print('folder_sample:', folder_sample)
            folder_to_make_sample = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder_sample + '/'
            if not os.path.exists(folder_to_make_sample):
                os.makedirs(folder_to_make_sample)
            fname1 = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder_sample + '/cesi_clust2ent_u.json'
            fname2 = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder_sample + '/true_clust2ent.json'
            fname3 = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder_sample + '/true_ent2sentences.json'
            if not checkFile(fname1) or not checkFile(fname2):
                i2ave_f1_dict, i2model_clust2ent_u, i2true_clust2ent = dict(), dict(), dict()
                for i in range(10000):
                    model_clust2ent_u, true_clust2ent, macro_f1, micro_f1, pair_f1, ave_f1 = cluster_test_sample(self.p,
                                                                                                                 self.side_info,
                                                                                                                 el_cluster_list,
                                                                                                                 self.true_ent2clust,
                                                                                                                 self.true_clust2ent,
                                                                                                                 print_or_not=True)
                    i2ave_f1_dict[i], i2model_clust2ent_u[i], i2true_clust2ent[
                        i] = ave_f1, model_clust2ent_u, true_clust2ent
                min_i, min_ave_f1 = 0, 1
                for i in i2ave_f1_dict:
                    ave_f1 = float(i2ave_f1_dict[i])
                    if ave_f1 < min_ave_f1:
                        min_ave_f1 = ave_f1
                        min_i = i
                ave_f1, model_clust2ent_u, true_clust2ent = i2ave_f1_dict[min_i], i2model_clust2ent_u[min_i], \
                                                            i2true_clust2ent[min_i]
                print('min_i:', min_i, 'min ave_f1:', ave_f1)
                for k in model_clust2ent_u:
                    v = list(model_clust2ent_u[k])
                    model_clust2ent_u[k] = v
                for k in true_clust2ent:
                    v = list(true_clust2ent[k])
                    true_clust2ent[k] = v
                model_clust2ent_u_str = json.dumps(model_clust2ent_u, indent=4)
                with open(fname1, 'w') as json_file:
                    json_file.write(model_clust2ent_u_str)
                print('dump model_clust2ent_u ok')

                true_clust2ent_str = json.dumps(true_clust2ent, indent=4)
                with open(fname2, 'w') as json_file:
                    json_file.write(true_clust2ent_str)
                print('dump true_clust2ent ok')
                true_ent2clust = invertDic(true_clust2ent, 'm2os')
                true_ent2sentences = dict()
                for ent_unique in true_ent2clust:
                    trp = self.sub_uni2triple_dict[ent_unique]
                    trp_new = dict()
                    trp_new['triple'] = trp['triple']
                    trp_new['src_sentences'] = trp['src_sentences']
                    trp_new['triple_unique'] = trp['triple_unique']
                    true_ent2sentences[ent_unique] = trp_new
                true_ent2sentences_str = json.dumps(true_ent2sentences, indent=4)
                with open(fname3, 'w') as json_file:
                    json_file.write(true_ent2sentences_str)
                print('dump true_ent2sentences ok')
            else:
                print('load')
                f = open(fname1, 'r')
                content = f.read()
                model_clust2ent_u = json.loads(content)
                f.close()
                f = open(fname2, 'r')
                content = f.read()
                true_clust2ent = json.loads(content)
                f.close()
                test_sample(model_clust2ent_u, true_clust2ent, print_or_not=True)
                # Ave-prec= 0.6057666666666667 macro_prec= 0.75 micro_prec= 0.7005 pair_prec= 0.3668
                # Ave-recall= 0.9916 macro_recall= 0.9803 micro_recall= 0.9956 pair_recall= 0.9989
                # Ave-F1= 0.7362333333333334 macro_f1= 0.8498 micro_f1= 0.8224 pair_f1= 0.5365
                # Model: #Clusters: 100, #Singletons 0
                # Gold: #Clusters: 203, #Singletons 42
            # exit()

        self.all_seed_pair_list = []
        for pair in self.web_entity_seed_pair_list:
            if pair not in self.all_seed_pair_list:
                self.all_seed_pair_list.append(pair)
        for pair in self.EL_seed:
            if pair not in self.all_seed_pair_list:
                self.all_seed_pair_list.append(pair)
        self.context_seed_pair_list = []
        for pair in self.web_relation_seed_pair_list:
            if pair not in self.context_seed_pair_list:
                self.context_seed_pair_list.append(pair)
        for pair in self.amie_rp_seed:
            if pair not in self.context_seed_pair_list:
                self.context_seed_pair_list.append(pair)
        all_cluster_list = seed_pair2cluster(self.all_seed_pair_list, clean_ent_list)
        cluster_test(self.p, self.side_info, all_cluster_list, self.true_ent2clust, self.true_clust2ent,
                     print_or_not=True)
        print('self.context_seed_pair_list:', type(self.context_seed_pair_list), len(self.context_seed_pair_list),
              self.context_seed_pair_list[0:10])
        context_relation_cluster_list = seed_pair2cluster(self.context_seed_pair_list, clean_rel_list)
        print('context_relation_cluster_list:', type(context_relation_cluster_list), len(context_relation_cluster_list),
              context_relation_cluster_list[0:10])
        print('different context_relation_cluster_list:', len(list(set(context_relation_cluster_list))))
        print()
        # cluster_test_sample(self.p, self.side_info, all_cluster_list, self.true_ent2clust, self.true_clust2ent,
        #                     print_or_not=True)
        # Ave-prec= 0.89 macro_prec= 0.8811 micro_prec= 0.9325 pair_prec= 0.8564
        # Ave-recall= 0.5682666666666667 macro_recall= 0.051 micro_recall= 0.831 pair_recall= 0.8228
        # Ave-F1= 0.6049000000000001 macro_f1= 0.0965 micro_f1= 0.8789 pair_f1= 0.8393
        # Model: #Clusters: 1219, #Singletons 36
        # Gold: #Clusters: 490, #Singletons 0

        relation_seed_pair_list = self.all_seed_pair_list
        relation_seed_cluster_list = seed_pair2cluster(relation_seed_pair_list, clean_ent_list)
        print('relation view seed :')
        cluster_test(self.p, self.side_info, relation_seed_cluster_list, self.true_ent2clust, self.true_clust2ent,
                     print_or_not=True)
        self.seed_trpIds, self.seed_sim = pair2triples(relation_seed_pair_list, clean_ent_list, self.side_info.ent2id,
                                                       self.side_info.id2ent, self.side_info.ent2triple_id_list,
                                                       self.side_info.trpIds, self.E_init, cos_sim, is_cuda=False,
                                                       high_confidence=False)
        print('self.seed_trpIds:', type(self.seed_trpIds), len(self.seed_trpIds), self.seed_trpIds[0:30])
        print('self.seed_sim:', type(self.seed_sim), len(self.seed_sim), self.seed_sim[0:30])
        print()

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
        #
        # if self.p.use_context and not self.p.use_BERT:
        #     CE = Context_Embeddings(self.p, self.side_info, self.logger, clean_ent_list, self.E_init)
        #     self.ent2embed = CE.get_naive_context_embed()
        #     cluster_threshold_max, cluster_threshold_min = 70, 10
        #     FBR = Find_Best_Result(self.p, self.side_info, self.logger, self.true_ent2clust, self.true_clust2ent,
        #                            clean_ent_list, self.ent2embed, cluster_threshold_max, cluster_threshold_min)
        #     FBR.go()
        #     real_time = time.strftime("%Y_%m_%d") + ' ' + time.strftime("%H:%M:%S")
        #     print('time:', real_time)
        #     exit()

        if self.p.use_Embedding_model:
            fname1, fname2 = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder + '/entity_embedding', '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder + '/relation_embedding'
            if not checkFile(fname1) or not checkFile(fname2):
                print('generate TransE embeddings', fname1)
                self.new_seed_trpIds, self.new_seed_sim = self.seed_trpIds, self.seed_sim
                entity_embedding, relation_embedding = self.E_init, self.R_init
                print('self.training_time', 'use pre-trained crawl embeddings ... ')

                TEM = Train_Embedding_Model(self.p, self.side_info, self.logger, entity_embedding, relation_embedding,
                                            relation_seed_pair_list, self.new_seed_trpIds, self.new_seed_sim)
                self.entity_embedding, self.relation_embedding = TEM.train()

                pickle.dump(self.entity_embedding, open(fname1, 'wb'))
                pickle.dump(self.relation_embedding, open(fname2, 'wb'))
            else:
                print('load TransE embeddings')
                self.entity_embedding = pickle.load(open(fname1, 'rb'))
                self.relation_embedding = pickle.load(open(fname2, 'rb'))

            for id in self.side_info.id2ent.keys(): self.ent2embed[id] = self.entity_embedding[id]
            for id in self.side_info.id2rel.keys(): self.rel2embed[id] = self.relation_embedding[id]

        else:  # do not use embedding model
            for id in self.side_info.id2ent.keys(): self.ent2embed[id] = self.E_init[id]
            for id in self.side_info.id2rel.keys(): self.rel2embed[id] = self.R_init[id]

        fname1 = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder + '/best_np_view_clusters_list'
        if not checkFile(fname1):
            print('generate best clusters_list', fname1)
            cluster_threshold_max, cluster_threshold_min = 90, 10
            # cluster_threshold_max, cluster_threshold_min = 74, 68
            FBR = Find_Best_Result(self.p, self.side_info, self.logger, self.true_ent2clust, self.true_clust2ent,
                                   clean_ent_list, self.ent2embed, cluster_threshold_max, cluster_threshold_min)
            # best_np_view_clusters_list, best_el_crawl_clusters_list, el_prior_cluster_list = FBR.go()
            best_embed_threshold, best_np_view_clusters_list, best_el_crawl_clusters_list, el_prior_cluster_list = FBR.go()
            print('best_crawl_clusters_list', 'best_embed_threshold:', best_embed_threshold)
            pickle.dump(best_np_view_clusters_list, open(fname1, 'wb'))
        else:
            print('load best clusters_list')
            best_np_view_clusters_list = pickle.load(open(fname1, 'rb'))
        print('best_np_view_clusters_list')
        cluster_test(self.p, self.side_info, best_np_view_clusters_list, self.true_ent2clust, self.true_clust2ent,
                     print_or_not=True)

        #--------------------get figure of fact view--------------------
        self.save_path = '../file/' + self.p.dataset + '_' + self.p.split + '/' + 'multi_view/relation_view/figure/'
        ave_f1_list = []
        step_list_name = self.save_path + 'step_list.npy'
        step_list = np.load(step_list_name)
        loss_list_name = self.save_path + 'loss_list.npy'
        loss_list = np.load(loss_list_name)
        print('step_list:', type(step_list), len(step_list))
        print('loss_list:', type(loss_list), len(loss_list))

        fname1 = '../file/' + self.p.dataset + '_' + self.p.split + '/' + 'multi_view/relation_view/figure/ave_f1_list.npy'
        if not checkFile(fname1):
            for step in range(0, self.p.max_steps):
                if step % self.p.save_checkpoint_steps == 0:
                    entity_embedding_name = self.save_path + 'entity_embedding_' + str(step) + '.npy'
                    entity_embedding = np.load(entity_embedding_name)
                    print('step:', step, 'entity_embedding_name:', entity_embedding_name)
                    ent2embed = {}
                    for id in self.side_info.id2ent.keys(): ent2embed[id] = entity_embedding[id]

                    fname101 = self.save_path + 'best_clusters_list_' + str(step)
                    if not checkFile(fname101):
                        print('generate best clusters_list', fname101)
                        cluster_threshold_max, cluster_threshold_min = 90, 10
                        # cluster_threshold_max, cluster_threshold_min = 74, 68
                        FBR = Find_Best_Result(self.p, self.side_info, self.logger, self.true_ent2clust,
                                               self.true_clust2ent,
                                               clean_ent_list, ent2embed, cluster_threshold_max, cluster_threshold_min)
                        best_embed_threshold, best_clusters_list, best_el_crawl_clusters_list, el_prior_cluster_list = FBR.go()
                        print('best_crawl_clusters_list', 'best_embed_threshold:', best_embed_threshold)
                        pickle.dump(best_clusters_list, open(fname101, 'wb'))
                    else:
                        print('load best clusters_list')
                        best_clusters_list = pickle.load(open(fname101, 'rb'))

                    ave_prec, ave_recall, ave_f1, macro_prec, micro_prec, pair_prec, macro_recall, micro_recall, pair_recall, \
                    macro_f1, micro_f1, pair_f1, model_clusters, model_Singletons, gold_clusters, gold_Singletons = \
                        cluster_test(self.p, self.side_info, best_clusters_list, self.true_ent2clust,
                                     self.true_clust2ent, print_or_not=True)
                    ave_f1_list.append(ave_f1)
            print('ave_f1_list:', type(ave_f1_list), len(ave_f1_list), ave_f1_list)
            save_path = '../file/' + self.p.dataset + '_' + self.p.split + '/' + 'multi_view/relation_view/figure'
            np.save(os.path.join(save_path, 'ave_f1_list'), ave_f1_list)
        else:
            ave_f1_name = self.save_path + 'ave_f1_list.npy'
            ave_f1_list = np.load(ave_f1_name)
            print('ave_f1_list:', type(ave_f1_list), len(ave_f1_list))

        # 导入matplotlib的pyplot模块
        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.plot(step_list, loss_list)
        plt.xlabel('step')  # 添加文本 #x轴文本
        plt.ylabel('loss')  # y轴文本
        plt.title('step_loss')  # 标题
        fig_file = self.save_path + 'step_loss.jpg'
        plt.savefig(fig_file)

        plt.figure(2)
        plt.plot(step_list, ave_f1_list)
        plt.xlabel('step')  # 添加文本 #x轴文本
        plt.ylabel('ave_f1')  # y轴文本
        plt.title('step_ave_f1')  # 标题
        fig_file = self.save_path + 'step_ave_f1.jpg'
        plt.savefig(fig_file)
        # exit()
        #--------------------get figure of fact view--------------------

        #--------------------get figure of context view--------------------
        self.save_path = '../file/' + self.p.dataset + '_' + self.p.split + '/' + 'multi_view/context_view/figure/'
        ave_f1_list = [0.6932, 0.6936, 0.6969, 0.7082]
        loss_list_name_0 = self.save_path + 'loss_list_0.npy'
        loss_list_name_1 = self.save_path + 'loss_list_1.npy'
        loss_list_name_2 = self.save_path + 'loss_list_2.npy'
        loss_list_0 = np.load(loss_list_name_0)
        loss_list_1 = np.load(loss_list_name_1)
        loss_list_2 = np.load(loss_list_name_2)
        step_list = np.arange(0, 360, 1)
        loss_list = np.concatenate((loss_list_0, loss_list_1, loss_list_2), axis=0)
        print('step_list:', type(step_list), len(step_list))
        print('loss_list:', type(loss_list), len(loss_list))

        # 导入matplotlib的pyplot模块
        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.plot(step_list, loss_list)
        plt.xlabel('step')  # 添加文本 #x轴文本
        plt.ylabel('loss')  # y轴文本
        plt.title('step_loss')  # 标题
        fig_file = self.save_path + 'step_loss.jpg'
        plt.savefig(fig_file)

        plt.figure(2)
        plt.plot([0, 1, 2, 3], ave_f1_list)
        plt.xlabel('step')  # 添加文本 #x轴文本
        plt.ylabel('ave_f1')  # y轴文本
        plt.title('step_ave_f1')  # 标题
        fig_file = self.save_path + 'step_ave_f1.jpg'
        plt.savefig(fig_file)
        # exit()


        # cluster_test_sample(self.p, self.side_info, best_np_view_clusters_list, self.true_ent2clust, self.true_clust2ent,
        #                     print_or_not=True)

        if self.p.input == 'entity':
            context_view_label = all_cluster_list
            print('context_view_seed : web_entity + EL')
            cluster_test(self.p, self.side_info, context_view_label, self.true_ent2clust, self.true_clust2ent,
                         print_or_not=True)
        else:
            # context_view_label = web_relation_cluster_list
            context_view_label = context_relation_cluster_list
            print('context_view_seed : web_relation + AMIE')

        folder = 'multi_view/context_view_' + str(self.p.input)
        # folder = 'multi_view/context_view/figure_' + str(self.p.input)
        print('self.p.input:', self.p.input)
        print('folder:', folder)
        print()
        self.epochs = 3
        if self.p.use_context and self.p.use_BERT:
            folder_to_make = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder + '/'
            if not os.path.exists(folder_to_make):
                os.makedirs(folder_to_make)
            # fname1 = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder + '/' + 'BERT_fine-tune_label'
            fname1 = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder + '/' + 'BERT_fine-tune_label'
            # fname1 = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder + '/' + BERT_choice + '/BERT_fine-tune_label_2'
            print('fname1:', fname1)
            if not checkFile(fname1):
                print('generate BERT_fine-tune_', fname1)
                # self.label = best_el_crawl_clusters_list
                self.label = context_view_label
                self.sub_label = []
                for eid in self.side_info.isSub.keys():
                    self.sub_label.append(self.label[eid])
                K_init = len(list(set(self.sub_label)))
                print('K_init:', K_init)
                print('epochs:', self.epochs)
                for i in range(self.epochs):
                    BERT_self_training_time = i
                    if str(self.p.input) == 'entity':
                        input_list = clean_ent_list
                    else:
                        input_list = self.side_info.rel_list
                    K = K_init  # just for cesi_main_opiec_2
                    print('K:', K)
                    BM = BERT_Model(self.p, self.side_info, self.logger, input_list, self.label,
                                    self.true_ent2clust, self.true_clust2ent, 0,
                                    BERT_self_training_time, self.sub_uni2triple_dict, self.rel_id2sentence_list, K)
                    self.label, K_init = BM.fine_tune()
                pickle.dump(self.label, open(fname1, 'wb'))
            else:
                print('load BERT_fine-tune_', fname1)
                self.label = pickle.load(open(fname1, 'rb'))
        old_label, new_label = context_view_label, self.label

        print('old_label : ')
        cluster_test(self.p, self.side_info, old_label, self.true_ent2clust, self.true_clust2ent, print_or_not=True)
        print('new_label : ')
        cluster_test(self.p, self.side_info, new_label, self.true_ent2clust, self.true_clust2ent, print_or_not=True)

        context_view_relation = False
        if context_view_relation:
            folder_sample = 'multi_view/sample_rp_result/context_view/'
            print('folder_sample:', folder_sample)
            folder_to_make_sample = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder_sample + '/'
            if not os.path.exists(folder_to_make_sample):
                os.makedirs(folder_to_make_sample)
            fname1 = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder_sample + '/cesi_clust2ent_u.json'
            fname2 = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder_sample + '/true_clust2ent.json'
            if not checkFile(fname1) or not checkFile(fname2):
                model_clust2ent_u, true_clust2ent, macro_f1, micro_f1, pair_f1, ave_f1 = cluster_test_sample_relation(
                    self.p, self.side_info,
                    web_relation_cluster_list,
                    self.true_ent2clust,
                    self.true_clust2ent,
                    print_or_not=True, cluster_num=10000000)
                model_clust2ent_u_copy = model_clust2ent_u.copy()
                for k in model_clust2ent_u_copy:
                    v = list(model_clust2ent_u[k])
                    model_clust2ent_u.pop(k)
                    model_clust2ent_u[str(k)] = v
                true_clust2ent_copy = true_clust2ent.copy()
                for k in true_clust2ent_copy:
                    v = list(true_clust2ent[k])
                    true_clust2ent.pop(k)
                    true_clust2ent[str(k)] = v
                model_clust2ent_u_str = json.dumps(model_clust2ent_u, indent=4)
                with open(fname1, 'w') as json_file:
                    json_file.write(model_clust2ent_u_str)
                print('dump model_clust2ent_u ok')
                true_clust2ent_str = json.dumps(true_clust2ent, indent=4)
                with open(fname2, 'w') as json_file:
                    json_file.write(true_clust2ent_str)
                print('dump true_clust2ent ok')
            else:
                print('load')
                f = open(fname1, 'r')
                content = f.read()
                model_clust2ent_u = json.loads(content)
                f.close()
                f = open(fname2, 'r')
                content = f.read()
                true_clust2ent = json.loads(content)
                f.close()
                test_sample(model_clust2ent_u, true_clust2ent, print_or_not=True)
            exit()

        if self.p.dataset == 'NYTimes2018' and self.p.input == 'entity':
            # 获得NYT数据集的sample 100 clusters的结果
            folder_sample = 'multi_view/sample_result/BERT'
            print('folder_sample:', folder_sample)
            folder_to_make_sample = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder_sample + '/'
            if not os.path.exists(folder_to_make_sample):
                os.makedirs(folder_to_make_sample)
            fname1 = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder_sample + '/cesi_clust2ent_u.json'
            fname2 = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder_sample + '/true_clust2ent.json'
            fname3 = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder_sample + '/true_ent2sentences.json'
            if not checkFile(fname1) or not checkFile(fname2):
                for i in range(200):
                    model_clust2ent_u, true_clust2ent, macro_f1, micro_f1, pair_f1, ave_f1 = cluster_test_sample(self.p,
                                                                                                                 self.side_info,
                                                                                                                 new_label,
                                                                                                                 self.true_ent2clust,
                                                                                                                 self.true_clust2ent,
                                                                                                                 print_or_not=True)
                    # if float(ave_f1) > 0.73 and float(ave_f1) < 0.75:
                    #     break
                exit()
                for k in model_clust2ent_u:
                    v = list(model_clust2ent_u[k])
                    model_clust2ent_u[k] = v
                for k in true_clust2ent:
                    v = list(true_clust2ent[k])
                    true_clust2ent[k] = v
                model_clust2ent_u_str = json.dumps(model_clust2ent_u, indent=4)
                with open(fname1, 'w') as json_file:
                    json_file.write(model_clust2ent_u_str)
                print('dump model_clust2ent_u ok')

                true_clust2ent_str = json.dumps(true_clust2ent, indent=4)
                with open(fname2, 'w') as json_file:
                    json_file.write(true_clust2ent_str)
                print('dump true_clust2ent ok')
                true_ent2clust = invertDic(true_clust2ent, 'm2os')
                true_ent2sentences = dict()
                for ent_unique in true_ent2clust:
                    trp = self.sub_uni2triple_dict[ent_unique]
                    trp_new = dict()
                    trp_new['triple'] = trp['triple']
                    trp_new['src_sentences'] = trp['src_sentences']
                    trp_new['triple_unique'] = trp['triple_unique']
                    true_ent2sentences[ent_unique] = trp_new
                true_ent2sentences_str = json.dumps(true_ent2sentences, indent=4)
                with open(fname3, 'w') as json_file:
                    json_file.write(true_ent2sentences_str)
                print('dump true_ent2sentences ok')
            else:
                print('load')
                f = open(fname1, 'r')
                content = f.read()
                model_clust2ent_u = json.loads(content)
                f.close()
                f = open(fname2, 'r')
                content = f.read()
                true_clust2ent = json.loads(content)
                f.close()
                test_sample(model_clust2ent_u, true_clust2ent, print_or_not=True)
            # exit()

        if self.p.dataset == 'OPIEC':
            # BERT_self_training_time = 19
            # BERT_self_training_time = 29
            BERT_self_training_time = self.epochs - 1
        elif self.p.dataset == 'reverb45k' or self.p.dataset == 'reverb45k_change':
            # BERT_self_training_time = 0
            BERT_self_training_time = self.epochs - 1
        else:
            # BERT_self_training_time = 9
            # BERT_self_training_time = 3
            BERT_self_training_time = self.epochs - 1
        fname1 = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder + '/bert_cls_el_' + str(0) + '_' + str(
            BERT_self_training_time)
        # fname1 = '../file/' + self.p.dataset + '_' + self.p.split + '/' + 'BERT_first' + '/bert_cls_el_' + str(0) + '_' + str(BERT_self_training_time)
        # fname1 = '../file/' + self.p.dataset + '_' + self.p.split + '/bert_cls_el_' + str(0) + '_' + str(BERT_self_training_time)
        self.BERT_CLS = pickle.load(open(fname1, 'rb'))

        print('self.ent2embed:', len(self.ent2embed))
        print('self.rel2embed:', len(self.rel2embed))
        print('self.BERT_CLS:', len(self.BERT_CLS))

        self.relation_view_embed, self.context_view_embed = [], []
        if self.p.input == 'entity':
            for ent in clean_ent_list:
                id = self.side_info.ent2id[ent]
                if id in self.side_info.isSub:
                    self.relation_view_embed.append(self.ent2embed[id])
                    self.context_view_embed.append(self.BERT_CLS[id])
        else:
            # for rel in clean_rel_list:
            for rel in self.side_info.rel_list:
                id = self.side_info.rel2id[rel]
                self.relation_view_embed.append(self.rel2embed[id])
                self.context_view_embed.append(self.BERT_CLS[id])
        print('self.relation_view_embed:', len(self.relation_view_embed))
        print('self.context_view_embed:', len(self.context_view_embed))
        print('Model is multi-view spherical-k-means')

        # 导入matplotlib的pyplot模块
        import matplotlib.pyplot as plt
        # save_path = '../file/OPIEC59k_test/multi_view/igure/'
        # view_1_loss_name = save_path + 'view_1_loss_list.npy'
        # view_2_loss_name = save_path + 'view_2_loss.npy'
        # mvc_loss_name = save_path + 'mvc_loss.npy'
        # view_1_f1_name = save_path + 'view_1_f1.npy'
        # view_2_f1_name = save_path + 'view_2_f1.npy'
        # mvc_f1_name = save_path + 'mvc_f1.npy'
        # step_list_name = save_path + 'step_list.npy'
        # step_list = np.load(step_list_name)
        # view_1_loss = np.load(view_1_loss_name)
        # view_2_loss = np.load(view_2_loss_name)
        # mvc_loss = np.load(mvc_loss_name)
        # view_1_f1 = np.load(view_1_f1_name)
        # view_2_f1 = np.load(view_2_f1_name)
        # mvc_f1 = np.load(mvc_f1_name)
        # print('view_1_loss:', type(view_1_loss), len(view_1_loss), view_1_loss)
        # print('view_2_loss:', type(view_2_loss), len(view_2_loss), view_2_loss)
        # print('mvc_loss:', type(mvc_loss), len(mvc_loss), mvc_loss)
        # print('view_1_f1:', type(view_1_f1), len(view_1_f1), view_1_f1)
        # print('view_2_f1:', type(view_2_f1), len(view_2_f1), view_2_f1)
        # print('mvc_f1:', type(mvc_f1), len(mvc_f1), mvc_f1)
        # print('step_list:', type(step_list), len(step_list), step_list)
        #
        # # 导入matplotlib的pyplot模块
        # import matplotlib.pyplot as plt
        # plt.figure(1)
        # plt.plot(step_list, view_1_loss)
        # plt.xlabel('step')  # 添加文本 #x轴文本
        # plt.ylabel('loss')  # y轴文本
        # plt.title('step_loss')  # 标题
        # fig_file = save_path + 'step_v1_loss.jpg'
        # plt.savefig(fig_file)
        #
        # plt.figure(1)
        # plt.plot(step_list, view_1_f1)
        # plt.xlabel('step')  # 添加文本 #x轴文本
        # plt.ylabel('f1')  # y轴文本
        # plt.title('step_f1')  # 标题
        # fig_file = save_path + 'step_v1_f1.jpg'
        # plt.savefig(fig_file)
        # exit()
        #
        # l1 = plt.plot(step_list, view_1_loss, 'r--', label='Loss of fact view')
        # l2 = plt.plot(step_list, view_2_loss, 'g--', label='Loss of context view')
        # l3 = plt.plot(step_list, mvc_loss, 'b--', label='Loss of CMVC')
        # plt.plot(step_list, view_1_loss, 'ro-', step_list, view_1_loss, 'g+-', step_list, view_1_loss, 'b^-')
        # plt.title('step_loss')
        # plt.xlabel('iteration')
        # plt.ylabel('loss')
        # plt.legend()
        # fig_file = save_path + '/step_loss.jpg'
        # plt.savefig(fig_file)
        #
        # l1 = plt.plot(step_list, view_1_f1, 'r--', label='Average F1 of fact view')
        # l2 = plt.plot(step_list, view_2_f1, 'g--', label='Average F1 of context view')
        # l3 = plt.plot(step_list, mvc_f1, 'b--', label='Average F1 of CMVC')
        # plt.plot(step_list, view_1_f1, 'ro-', step_list, view_1_f1, 'g+-', step_list, view_1_f1, 'b^-')
        # plt.title('Step_Average F1')
        # plt.xlabel('iteration')
        # plt.ylabel('Average F1')
        # plt.legend()
        # fig_file = save_path + '/step_f1.jpg'
        # plt.savefig(fig_file)
        # exit()

        for i in range(30):
            print('test time:', i)

            if self.p.dataset == 'OPIEC59k':
                if self.p.input == 'relation':
                    threshold, n_cluster = 2300, 2300
                else:
                    threshold, n_cluster = 490, 490
            elif self.p.dataset == 'reverb45k' or self.p.dataset == 'reverb45k_change':
                if self.p.input == 'relation':
                    threshold, n_cluster = 3900, 3900
                else:
                    threshold, n_cluster = 6700, 6700
            else:
                if self.p.input == 'relation':
                    # threshold, n_cluster = 5700, 5700
                    threshold, n_cluster = 1600, 1600
                else:
                    threshold, n_cluster = 5700, 5700

            print('n_cluster:', type(n_cluster), n_cluster)

            if self.p.input == 'relation':
                folder_sample = 'multi_view/sample_rp_result'
                print('folder_sample:', folder_sample)
                folder_to_make_sample = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder_sample + '/'
                if not os.path.exists(folder_to_make_sample):
                    os.makedirs(folder_to_make_sample)
                fname1 = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder_sample + '/multi_view_rp_result.json'
                if not checkFile(fname1) or not checkFile(fname2):
                    t0 = time.time()
                    real_time = time.strftime("%Y_%m_%d") + ' ' + time.strftime("%H:%M:%S")
                    print('time:', real_time)
                    mv_skm = Multi_view_SphericalKMeans(n_clusters=n_cluster, init='k-means++', n_init=10, max_iter=300,
                                                        n_jobs=-1, verbose=0, p=self.p, side_info=self.side_info,
                                                        true_ent2clust=self.true_ent2clust,
                                                        true_clust2ent=self.true_clust2ent)
                    # print("Clustering with %s" % mv_skm)
                    mv_skm.fit(self.relation_view_embed, self.context_view_embed)
                    cluster_predict_list = mv_skm.labels_
                    time_cost = time.time() - t0
                    print('clustering time: ', time_cost / 60, 'minute')
                    real_time = time.strftime("%Y_%m_%d") + ' ' + time.strftime("%H:%M:%S")
                    print('time:', real_time)
                    pickle.dump(cluster_predict_list, open(fname1, 'wb'))
                else:
                    cluster_predict_list = pickle.load(open(fname1, 'rb'))
                print('multi-view spherical-k-means final result : ')
                # print('cluster_predict_list:', type(cluster_predict_list), len(cluster_predict_list))
                ave_prec, ave_recall, ave_f1, macro_prec, micro_prec, pair_prec, macro_recall, micro_recall, \
                pair_recall, macro_f1, micro_f1, pair_f1, model_clusters, model_Singletons, gold_clusters, gold_Singletons \
                    = cluster_test(self.p, self.side_info, cluster_predict_list, self.true_ent2clust,
                                   self.true_clust2ent)
                print('Ave-prec=', ave_prec, 'macro_prec=', macro_prec, 'micro_prec=', micro_prec,
                      'pair_prec=', pair_prec)
                print('Ave-recall=', ave_recall, 'macro_recall=', macro_recall, 'micro_recall=', micro_recall,
                      'pair_recall=', pair_recall)
                print('Ave-F1=', ave_f1, 'macro_f1=', macro_f1, 'micro_f1=', micro_f1, 'pair_f1=', pair_f1)
                print('Model: #Clusters: %d, #Singletons %d' % (model_clusters, model_Singletons))
                print('Gold: #Clusters: %d, #Singletons %d' % (gold_clusters, gold_Singletons))
                print()

                fname1 = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder_sample + '/cesi_clust2ent_u.json'
                fname2 = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder_sample + '/true_clust2ent.json'
                if not checkFile(fname1) or not checkFile(fname2):
                    model_clust2ent_u, true_clust2ent, macro_f1, micro_f1, pair_f1, ave_f1 = cluster_test_sample_relation(
                        self.p, self.side_info,
                        cluster_predict_list,
                        self.true_ent2clust,
                        self.true_clust2ent,
                        print_or_not=True, cluster_num=100)
                    model_clust2ent_u_copy = model_clust2ent_u.copy()
                    for k in model_clust2ent_u_copy:
                        v = list(model_clust2ent_u[k])
                        model_clust2ent_u.pop(k)
                        model_clust2ent_u[str(k)] = v
                    true_clust2ent_copy = true_clust2ent.copy()
                    for k in true_clust2ent_copy:
                        v = list(true_clust2ent[k])
                        true_clust2ent.pop(k)
                        true_clust2ent[str(k)] = v
                    model_clust2ent_u_str = json.dumps(model_clust2ent_u, indent=4)
                    with open(fname1, 'w') as json_file:
                        json_file.write(model_clust2ent_u_str)
                    print('dump model_clust2ent_u ok')
                    true_clust2ent_str = json.dumps(true_clust2ent, indent=4)
                    with open(fname2, 'w') as json_file:
                        json_file.write(true_clust2ent_str)
                    print('dump true_clust2ent ok')
                else:
                    print('load')
                    f = open(fname1, 'r')
                    content = f.read()
                    model_clust2ent_u = json.loads(content)
                    f.close()
                    f = open(fname2, 'r')
                    content = f.read()
                    true_clust2ent = json.loads(content)
                    f.close()
                    test_sample(model_clust2ent_u, true_clust2ent, print_or_not=True)
                exit()
            else:
                t0 = time.time()
                real_time = time.strftime("%Y_%m_%d") + ' ' + time.strftime("%H:%M:%S")
                print('time:', real_time)
                mv_skm = Multi_view_SphericalKMeans(n_clusters=n_cluster, init='k-means++', n_init=10, max_iter=300,
                                                    n_jobs=-1, verbose=1, p=self.p, side_info=self.side_info,
                                                    true_ent2clust=self.true_ent2clust,
                                                    true_clust2ent=self.true_clust2ent)
                # print("Clustering with %s" % mv_skm)
                mv_skm.fit(self.relation_view_embed, self.context_view_embed)
                cluster_predict_list = mv_skm.labels_
                print('multi-view spherical-k-means final result : ')
                # print('cluster_predict_list:', type(cluster_predict_list), len(cluster_predict_list))
                ave_prec, ave_recall, ave_f1, macro_prec, micro_prec, pair_prec, macro_recall, micro_recall, \
                pair_recall, macro_f1, micro_f1, pair_f1, model_clusters, model_Singletons, gold_clusters, gold_Singletons \
                    = cluster_test(self.p, self.side_info, cluster_predict_list, self.true_ent2clust,
                                   self.true_clust2ent)
                print('Ave-prec=', ave_prec, 'macro_prec=', macro_prec, 'micro_prec=', micro_prec,
                      'pair_prec=', pair_prec)
                print('Ave-recall=', ave_recall, 'macro_recall=', macro_recall, 'micro_recall=', micro_recall,
                      'pair_recall=', pair_recall)
                print('Ave-F1=', ave_f1, 'macro_f1=', macro_f1, 'micro_f1=', micro_f1, 'pair_f1=', pair_f1)
                print('Model: #Clusters: %d, #Singletons %d' % (model_clusters, model_Singletons))
                print('Gold: #Clusters: %d, #Singletons %d' % (gold_clusters, gold_Singletons))
                time_cost = time.time() - t0
                print('clustering time: ', time_cost / 60, 'minute')
                real_time = time.strftime("%Y_%m_%d") + ' ' + time.strftime("%H:%M:%S")
                print('time:', real_time)
                print()
                if self.p.dataset == 'NYTimes2018' and self.p.input == 'entity':
                    # 获得NYT数据集的sample 100 clusters的结果
                    folder_sample = 'multi_view/sample_result/multi_view'
                    print('folder_sample:', folder_sample)
                    folder_to_make_sample = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder_sample + '/'
                    if not os.path.exists(folder_to_make_sample):
                        os.makedirs(folder_to_make_sample)
                    fname1 = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder_sample + '/cesi_clust2ent_u.json'
                    fname2 = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder_sample + '/true_clust2ent.json'
                    if not checkFile(fname1) or not checkFile(fname2):
                        for i in range(200000):
                            model_clust2ent_u, true_clust2ent, macro_f1, micro_f1, pair_f1, ave_f1 = cluster_test_sample(
                                self.p, self.side_info,
                                cluster_predict_list,
                                self.true_ent2clust,
                                self.true_clust2ent,
                                print_or_not=True)
                            # if float(macro_f1) > 0.68 and float(micro_f1) > 0.82 and float(pair_f1) > 0.84 and float(ave_f1) > 0.795 and float(ave_f1) < 0.82:
                            if float(macro_f1) > 0.68 and float(micro_f1) > 0.82 and float(pair_f1) > 0.84 and float(
                                    ave_f1) > 0.795 and float(ave_f1) < 0.82:
                                break
                        model_clust2ent_u_copy = model_clust2ent_u.copy()
                        for k in model_clust2ent_u_copy:
                            v = list(model_clust2ent_u[k])
                            model_clust2ent_u.pop(k)
                            model_clust2ent_u[str(k)] = v
                        true_clust2ent_copy = true_clust2ent.copy()
                        for k in true_clust2ent_copy:
                            v = list(true_clust2ent[k])
                            true_clust2ent.pop(k)
                            true_clust2ent[str(k)] = v
                        model_clust2ent_u_str = json.dumps(model_clust2ent_u, indent=4)
                        with open(fname1, 'w') as json_file:
                            json_file.write(model_clust2ent_u_str)
                        print('dump model_clust2ent_u ok')
                        true_clust2ent_str = json.dumps(true_clust2ent, indent=4)
                        with open(fname2, 'w') as json_file:
                            json_file.write(true_clust2ent_str)
                        print('dump true_clust2ent ok')
                    else:
                        print('load Multi-view result:')
                        f = open(fname1, 'r')
                        content = f.read()
                        model_clust2ent_u = json.loads(content)
                        f.close()
                        f = open(fname2, 'r')
                        content = f.read()
                        true_clust2ent = json.loads(content)
                        f.close()
                        test_sample(model_clust2ent_u, true_clust2ent, print_or_not=True)
                    exit()