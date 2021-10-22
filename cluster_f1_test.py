from helper import *
from utils import *
from metrics import evaluate  # Evaluation metrics
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from tqdm import tqdm
ave = True
# ave = False

def embed2f1(params, embed, cluster_threshold_real, side_info, true_ent2clust, true_clust2ent, dim_is_bert=False, print_or_not=True):
    clusters, clusters_center = HAC_getClusters(params, embed, cluster_threshold_real, dim_is_bert)
    n_clusters = int(clusters_center.shape[0])
    print('cluster_threshold_real:', cluster_threshold_real)
    if int(n_clusters) < 10:
        print('n_cluster is less than 10 ... so bad !')
        return 0
    else:
        cluster_predict_list = list(clusters)
        ave_prec, ave_recall, ave_f1, macro_prec, micro_prec, pair_prec, macro_recall, micro_recall, \
        pair_recall, macro_f1, micro_f1, pair_f1, model_clusters, model_Singletons, gold_clusters, gold_Singletons \
            = cluster_test(params, side_info, cluster_predict_list, true_ent2clust, true_clust2ent, print_or_not=print_or_not)
        return ave_f1

def HAC_getClusters(params, embed, cluster_threshold_real, dim_is_bert=False):
    if dim_is_bert:
        embed_dim = 768
    else:
        embed_dim = 300
    dist = pdist(embed, metric=params.metric)
    if params.dataset == 'reverb45k':
        if not np.all(np.isfinite(dist)):
            for i in range(len(dist)):
                if not np.isfinite(dist[i]):
                    dist[i] = 0
    clust_res = linkage(dist, method=params.linkage)
    labels = fcluster(clust_res, t=cluster_threshold_real, criterion='distance') - 1

    clusters = [[] for i in range(max(labels) + 1)]
    for i in range(len(labels)):
        clusters[labels[i]].append(i)

    clusters_center = np.zeros((len(clusters), embed_dim), np.float32)
    for i in range(len(clusters)):
        cluster = clusters[i]
        if ave:
            clusters_center_embed = np.zeros(embed_dim, np.float32)
            for j in cluster:
                embed_ = embed[j]
                clusters_center_embed += embed_
            clusters_center_embed_ = clusters_center_embed / len(cluster)
            clusters_center[i, :] = clusters_center_embed_
        else:
            sim_matrix = np.empty((len(cluster), len(cluster)), np.float32)
            for i in range(len(cluster)):
                for j in range(len(cluster)):
                    if i == j:
                        sim_matrix[i, j] = 1
                    else:
                        if params.metric == 'cosine':
                            sim = cos_sim(embed[i], embed[j])
                        else:
                            sim = np.linalg.norm(embed[i] - embed[j])
                        sim_matrix[i, j] = sim
                        sim_matrix[j, i] = sim
            sim_sum = sim_matrix.sum(axis=1)
            max_num = cluster[int(np.argmax(sim_sum))]
            clusters_center[i, :] = embed[max_num]
    # print('clusters_center:', type(clusters_center), clusters_center.shape)
    return labels, clusters_center


def cluster_test(params, side_info, cluster_predict_list, true_ent2clust, true_clust2ent, print_or_not=False, get_cluster_print=False):
    sub_cluster_predict_list = []
    clust2ent = {}
    isSub = side_info.isSub
    triples = side_info.triples
    ent2id = side_info.ent2id

    for eid in isSub.keys():
        sub_cluster_predict_list.append(cluster_predict_list[eid])

    for sub_id, cluster_id in enumerate(sub_cluster_predict_list):
        if cluster_id in clust2ent.keys():
            clust2ent[cluster_id].append(sub_id)
        else:
            clust2ent[cluster_id] = [sub_id]
    cesi_clust2ent = {}
    for rep, cluster in clust2ent.items():
        # cesi_clust2ent[rep] = list(cluster)
        cesi_clust2ent[rep] = set(cluster)
    cesi_ent2clust = invertDic(cesi_clust2ent, 'm2os')

    if get_cluster_print:
        fname = '../file/' + params.dataset + '/cesi_clust2ent'
        if not checkFile(fname):
            print('dump true_clust2ent')
            pickle.dump(cesi_clust2ent, open(fname, 'wb'))
        else:
            print('load true_clust2ent')
            cesi_clust2ent = pickle.load(open(fname, 'rb'))
        cesi_ent2clust = invertDic(cesi_clust2ent, 'm2os')
        print('cesi_clust2ent:', type(cesi_clust2ent), len(cesi_clust2ent))
        print('cesi_ent2clust:', type(cesi_ent2clust), len(cesi_ent2clust))
        i = 0
        for k in cesi_clust2ent:
            clust = cesi_clust2ent[k]
            if i < 10:
                print('i:', i, 'k:', k, 'clust:', clust)
            i += 1
        new_cesi_clust2ent = list(cesi_clust2ent.values())
        for i in range(len(new_cesi_clust2ent)):
            id_list = new_cesi_clust2ent[i]
            ent_list = []
            for id in id_list:
                ent = side_info.id2ent[id]
                ent_list.append(ent)
            new_cesi_clust2ent[i] = ent_list
        print('new_cesi_clust2ent:', type(new_cesi_clust2ent), len(new_cesi_clust2ent))
        print('请输入下次最大的i：')
        i_max = input()
        for i in range(len(new_cesi_clust2ent)):
            clust = new_cesi_clust2ent[i]
            if not i > int(i_max) + 1:
                print('i:', i, 'clust:', len(clust), clust)
        print('请输入要修改clust的i：')
        ask = input()
        start_new_ent_index = 0
        while ask != 'stop':
            clust = new_cesi_clust2ent[int(ask)]
            print('i:', ask, 'clust:', len(clust), clust)
            if len(clust) > 1:
                print('请输入要修改的ent：')
                ask_ent = input()
                while ask_ent not in clust:
                    print('上次输错了，请输入要修改的ent：')
                    ask_ent = input()
            else:
                ask_ent = clust[0]
            print('请输入要ent的新clust：(若要去新的clust则输入new)')
            ask_ent_new = input()
            print('ask_ent_new:', type(ask_ent_new), ask_ent_new)
            if ask_ent_new == 'new':
                if start_new_ent_index == 0:
                    start_new_ent_index = len(new_cesi_clust2ent)
                new_cesi_clust2ent.append([ask_ent])
            else:
                new_cesi_clust2ent[int(ask_ent_new)].append(ask_ent)
            clust.remove(ask_ent)
            print('修改后的i:', ask, 'clust:', len(clust), clust)

            i_max =  ask
            for i in range(len(new_cesi_clust2ent)):
                clust = new_cesi_clust2ent[i]
                if not i > int(i_max) + 1:
                    print('i:', i, 'clust:', len(clust), clust)
            if ask_ent_new == 'new':
                ask_ent_new = len(new_cesi_clust2ent) - 1
            print('修改后的ask_ent_new:', ask_ent_new, '修改后的new_clust:',
                      new_cesi_clust2ent[int(ask_ent_new)])
            print('new_cesi_clust2ent:', type(new_cesi_clust2ent), len(new_cesi_clust2ent))
            print('请输入要修改clust的i：')
            ask = input()
        print('stop ! 保存！')
        for i in range(len(new_cesi_clust2ent)):
            ent_list = new_cesi_clust2ent[i]
            id_list = []
            for ent in ent_list:
                id = side_info.ent2id[ent]
                id_list.append(id)
            new_cesi_clust2ent[i] = id_list
        j = 0
        for k in cesi_clust2ent:
            clust = new_cesi_clust2ent[j]
            cesi_clust2ent[k] = clust
            j += 1
        if start_new_ent_index > 0:
            num = start_new_ent_index
            for i in range(len(new_cesi_clust2ent[start_new_ent_index:])):
                new_ent = new_cesi_clust2ent[start_new_ent_index:][i]
                new_ent_int = new_ent[0]
                if new_ent_int not in cesi_clust2ent:
                    cesi_clust2ent.update({new_ent_int: set(new_ent)})
                    print('new_ent_int1:', type(new_ent_int), new_ent_int, 'new_ent:', new_ent)
                else:
                    print('new_ent_int2:', type(new_ent_int), new_ent_int, 'new_ent:', new_ent)
                    while new_ent_int in cesi_clust2ent:
                        new_ent_int += 10000
                    cesi_clust2ent.update({new_ent_int: set(new_ent)})
                    print('new_ent_int2:', type(new_ent_int), new_ent_int, 'new_ent:', new_ent)
                    num += 1
        cesi_ent2clust = invertDic(cesi_clust2ent, 'm2os')
        print('cesi_clust2ent:', type(cesi_clust2ent), len(cesi_clust2ent))
        print('cesi_ent2clust:', type(cesi_ent2clust), len(cesi_ent2clust))
        if len(cesi_ent2clust) != 7486:
            print('wrong ! ')
            exit()
        cesi_clust2ent_copy = cesi_clust2ent.copy()
        for k in cesi_clust2ent_copy:
            clust = cesi_clust2ent[k]
            if len(clust) < 1:
                cesi_clust2ent.pop(k)
        print('cesi_clust2ent:', type(cesi_clust2ent), len(cesi_clust2ent))
        print('dump true_clust2ent')
        pickle.dump(cesi_clust2ent, open(fname, 'wb'))
        print('load true_clust2ent')
        cesi_clust2ent = pickle.load(open(fname, 'rb'))
        cesi_ent2clust = invertDic(cesi_clust2ent, 'm2os')
        print('cesi_clust2ent:', type(cesi_clust2ent), len(cesi_clust2ent))
        print('cesi_ent2clust:', type(cesi_ent2clust), len(cesi_ent2clust))

        exit()

    cesi_ent2clust_u = {}
    if params.use_assume:
        for trp in triples:
            sub_u, sub = trp['triple_unique'][0], trp['triple'][0]
            cesi_ent2clust_u[sub_u] = cesi_ent2clust[ent2id[sub]]
    else:
        for trp in triples:
            sub_u, sub = trp['triple_unique'][0], trp['triple_unique'][0]
            cesi_ent2clust_u[sub_u] = cesi_ent2clust[ent2id[sub]]

    cesi_clust2ent_u = invertDic(cesi_ent2clust_u, 'm2os')

    eval_results = evaluate(cesi_ent2clust_u, cesi_clust2ent_u, true_ent2clust, true_clust2ent)
    macro_prec, micro_prec, pair_prec = eval_results['macro_prec'], eval_results['micro_prec'], eval_results['pair_prec']
    macro_recall, micro_recall, pair_recall = eval_results['macro_recall'], eval_results['micro_recall'], eval_results['pair_recall']
    macro_f1, micro_f1, pair_f1 = eval_results['macro_f1'], eval_results['micro_f1'], eval_results['pair_f1']
    ave_prec = (macro_prec + micro_prec + pair_prec) / 3
    ave_recall = (macro_recall + micro_recall + pair_recall) / 3
    ave_f1 = (macro_f1 + micro_f1 + pair_f1) / 3
    model_clusters = len(cesi_clust2ent_u)
    model_Singletons = len([1 for _, clust in cesi_clust2ent_u.items() if len(clust) == 1])
    gold_clusters = len(true_clust2ent)
    gold_Singletons = len([1 for _, clust in true_clust2ent.items() if len(clust) == 1])
    if print_or_not:
        print('Ave-prec=', ave_prec, 'macro_prec=', macro_prec, 'micro_prec=', micro_prec,
              'pair_prec=', pair_prec)
        print('Ave-recall=', ave_recall, 'macro_recall=', macro_recall, 'micro_recall=', micro_recall,
              'pair_recall=', pair_recall)
        print('Ave-F1=', ave_f1, 'macro_f1=', macro_f1, 'micro_f1=', micro_f1, 'pair_f1=', pair_f1)
        print('Model: #Clusters: %d, #Singletons %d' % (model_clusters, model_Singletons))
        print('Gold: #Clusters: %d, #Singletons %d' % (gold_clusters, gold_Singletons))
        print()

    return ave_prec, ave_recall, ave_f1, macro_prec, micro_prec, pair_prec, macro_recall, micro_recall, pair_recall, \
           macro_f1, micro_f1, pair_f1, model_clusters, model_Singletons, gold_clusters, gold_Singletons

def print_first_10_dict(input, all=False):
    num = 0
    for k in input:
        v = input[k]
        print('k:', type(k), k)
        print('v:', type(v), len(v), v)
        num += 1
        if num > 10 and not all:
            break
    print()

def change_true_label(true_label):
    for k in true_label:
        v_list = true_label[k]
        print('k:', type(k), k)
        for i in range(len(v_list)):
            v = v_list[i]
            print('i:', i, 'v:', type(v), len(v), v)
        print()
        print('有没有问题？')
        answer = input()
        if answer == 'Y' or answer == 'y':
            print('请输入有问题的index:')
            answer_index = input()
            wrong = v_list[answer_index]
    print()


def cluster_test_sample(params, side_info, cluster_predict_list, true_ent2clust, true_clust2ent, print_or_not=False, cluster_num=100):
    sub_cluster_predict_list = []
    clust2ent = {}
    isSub = side_info.isSub
    triples = side_info.triples
    ent2id = side_info.ent2id

    for eid in isSub.keys():
        sub_cluster_predict_list.append(cluster_predict_list[eid])

    for sub_id, cluster_id in enumerate(sub_cluster_predict_list):
        if cluster_id in clust2ent.keys():
            clust2ent[cluster_id].append(sub_id)
        else:
            clust2ent[cluster_id] = [sub_id]
    cesi_clust2ent = {}
    for rep, cluster in clust2ent.items():
        cesi_clust2ent[rep] = list(cluster)
    cesi_ent2clust = invertDic(cesi_clust2ent, 'm2os')

    cesi_ent2clust_u = {}
    if params.use_assume:
        for trp in triples:
            sub_u, sub = trp['triple_unique'][0], trp['triple'][0]
            cesi_ent2clust_u[sub_u] = cesi_ent2clust[ent2id[sub]]
    else:
        for trp in triples:
            sub_u, sub = trp['triple_unique'][0], trp['triple_unique'][0]
            cesi_ent2clust_u[sub_u] = cesi_ent2clust[ent2id[sub]]

    cesi_clust2ent_u = invertDic(cesi_ent2clust_u, 'm2os')

    # print('cesi_ent2clust_u:', type(cesi_ent2clust_u), len(cesi_ent2clust_u))
    # print('cesi_clust2ent_u:', type(cesi_clust2ent_u), len(cesi_clust2ent_u))

    k_list = list(set(list(cesi_clust2ent_u.keys())))
    import random
    random.shuffle(k_list)
    # print('k_list:', type(k_list), len(k_list), k_list)
    # k_list = k_list[0:100]
    new_k_list = []
    for i in k_list:
        clust = cesi_clust2ent_u[i]
        if len(clust) > 1:
            new_k_list.append(i)
        if len(new_k_list) == cluster_num:
            break
    # print('new_k_list:', type(new_k_list), len(new_k_list), new_k_list)
    assert len(new_k_list) == cluster_num
    new_dict = dict()
    for k in new_k_list:
        # new_dict.update({k: cesi_clust2ent_u[k]})
        new_dict.update({str(k): cesi_clust2ent_u[k]})
    assert len(new_dict) == cluster_num
    cesi_clust2ent_u = new_dict
    # print('cesi_clust2ent_u:', type(cesi_clust2ent_u), len(cesi_clust2ent_u), cesi_clust2ent_u)
    cesi_ent2clust_u = invertDic(cesi_clust2ent_u, 'm2os')
    # print('cesi_ent2clust_u:', type(cesi_ent2clust_u), len(cesi_ent2clust_u), cesi_ent2clust_u)

    new_true_ent2clust = dict()
    for ent in cesi_ent2clust_u:
        new_true_ent2clust.update({str(ent): true_ent2clust[ent]})

    # true_ent2clust = new_true_ent2clust
    new_true_clust2ent = invertDic(new_true_ent2clust, 'm2os')
    # print('true_clust2ent:', type(true_clust2ent), len(true_clust2ent), true_clust2ent)
    # print('true_ent2clust:', type(true_ent2clust), len(true_ent2clust), true_ent2clust)
    # print_first_10_dict(true_clust2ent, all=True)

    eval_results = evaluate(cesi_ent2clust_u, cesi_clust2ent_u, new_true_ent2clust, new_true_clust2ent)
    macro_prec, micro_prec, pair_prec = eval_results['macro_prec'], eval_results['micro_prec'], eval_results[
        'pair_prec']
    macro_recall, micro_recall, pair_recall = eval_results['macro_recall'], eval_results['micro_recall'], eval_results[
        'pair_recall']
    macro_f1, micro_f1, pair_f1 = eval_results['macro_f1'], eval_results['micro_f1'], eval_results['pair_f1']
    ave_prec = (macro_prec + micro_prec + pair_prec) / 3
    ave_recall = (macro_recall + micro_recall + pair_recall) / 3
    ave_f1 = (macro_f1 + micro_f1 + pair_f1) / 3
    model_clusters = len(cesi_clust2ent_u)
    model_Singletons = len([1 for _, clust in cesi_clust2ent_u.items() if len(clust) == 1])
    gold_clusters = len(new_true_clust2ent)
    gold_Singletons = len([1 for _, clust in new_true_clust2ent.items() if len(clust) == 1])
    if print_or_not:
        print('Ave-prec=', ave_prec, 'macro_prec=', macro_prec, 'micro_prec=', micro_prec,
              'pair_prec=', pair_prec)
        print('Ave-recall=', ave_recall, 'macro_recall=', macro_recall, 'micro_recall=', micro_recall,
              'pair_recall=', pair_recall)
        print('Ave-F1=', ave_f1, 'macro_f1=', macro_f1, 'micro_f1=', micro_f1, 'pair_f1=', pair_f1)
        print('Model: #Clusters: %d, #Singletons %d' % (model_clusters, model_Singletons))
        print('Gold: #Clusters: %d, #Singletons %d' % (gold_clusters, gold_Singletons))
        print()

    return cesi_clust2ent_u, new_true_clust2ent, macro_f1, micro_f1, pair_f1, ave_f1

def cluster_test_sample_relation(params, side_info, cluster_predict_list, true_ent2clust, true_clust2ent, print_or_not=False, cluster_num=35):
    clust2ent = {}
    triples = side_info.triples
    rel2id = side_info.rel2id

    for sub_id, cluster_id in enumerate(cluster_predict_list):
        if cluster_id in clust2ent.keys():
            clust2ent[cluster_id].append(sub_id)
        else:
            clust2ent[cluster_id] = [sub_id]
    cesi_clust2ent = {}
    for rep, cluster in clust2ent.items():
        cesi_clust2ent[rep] = list(cluster)
    cesi_ent2clust = invertDic(cesi_clust2ent, 'm2os')

    cesi_ent2clust_u = {}
    for trp in triples:
        rel = trp['triple'][1]
        if rel2id[rel] in cesi_ent2clust:
            cesi_ent2clust_u[rel] = cesi_ent2clust[rel2id[rel]]

    cesi_clust2ent_u = invertDic(cesi_ent2clust_u, 'm2os')

    # print('cesi_ent2clust_u:', type(cesi_ent2clust_u), len(cesi_ent2clust_u))
    # print('cesi_clust2ent_u:', type(cesi_clust2ent_u), len(cesi_clust2ent_u))

    k_list = list(set(list(cesi_clust2ent_u.keys())))
    import random
    random.shuffle(k_list)
    # print('k_list:', type(k_list), len(k_list), k_list)
    # k_list = k_list[0:100]
    new_k_list = []
    for i in k_list:
        clust = cesi_clust2ent_u[i]
        # if len(clust) > 1:
        # if len(clust) > 1 and len(clust) < 10:
        if len(clust) > 1 and len(clust) < 5:
            new_k_list.append(i)
        if len(new_k_list) == cluster_num:
            break
    # print('new_k_list:', type(new_k_list), len(new_k_list), new_k_list)
    # assert len(new_k_list) == cluster_num
    new_dict = dict()
    for k in new_k_list:
        # new_dict.update({k: cesi_clust2ent_u[k]})
        new_dict.update({str(k): cesi_clust2ent_u[k]})
    # assert len(new_dict) == cluster_num
    cesi_clust2ent_u = new_dict
    # print('cesi_clust2ent_u:', type(cesi_clust2ent_u), len(cesi_clust2ent_u), cesi_clust2ent_u)
    cesi_ent2clust_u = invertDic(cesi_clust2ent_u, 'm2os')
    # print('cesi_ent2clust_u:', type(cesi_ent2clust_u), len(cesi_ent2clust_u), cesi_ent2clust_u)

    eval_results = evaluate(cesi_ent2clust_u, cesi_clust2ent_u, cesi_ent2clust_u, cesi_clust2ent_u)
    macro_prec, micro_prec, pair_prec = eval_results['macro_prec'], eval_results['micro_prec'], eval_results[
        'pair_prec']
    macro_recall, micro_recall, pair_recall = eval_results['macro_recall'], eval_results['micro_recall'], eval_results[
        'pair_recall']
    macro_f1, micro_f1, pair_f1 = eval_results['macro_f1'], eval_results['micro_f1'], eval_results['pair_f1']
    ave_prec = (macro_prec + micro_prec + pair_prec) / 3
    ave_recall = (macro_recall + micro_recall + pair_recall) / 3
    ave_f1 = (macro_f1 + micro_f1 + pair_f1) / 3
    model_clusters = len(cesi_clust2ent_u)
    model_Singletons = len([1 for _, clust in cesi_clust2ent_u.items() if len(clust) == 1])
    gold_clusters = len(cesi_clust2ent_u)
    gold_Singletons = len([1 for _, clust in cesi_clust2ent_u.items() if len(clust) == 1])
    if print_or_not:
        print('Ave-prec=', ave_prec, 'macro_prec=', macro_prec, 'micro_prec=', micro_prec,
              'pair_prec=', pair_prec)
        print('Ave-recall=', ave_recall, 'macro_recall=', macro_recall, 'micro_recall=', micro_recall,
              'pair_recall=', pair_recall)
        print('Ave-F1=', ave_f1, 'macro_f1=', macro_f1, 'micro_f1=', micro_f1, 'pair_f1=', pair_f1)
        print('Model: #Clusters: %d, #Singletons %d' % (model_clusters, model_Singletons))
        print('Gold: #Clusters: %d, #Singletons %d' % (gold_clusters, gold_Singletons))
        print()

    return cesi_clust2ent_u, cesi_clust2ent_u, macro_f1, micro_f1, pair_f1, ave_f1

def test_sample(model_clust2ent_u, true_clust2ent, print_or_not=False):
    model_ent2clust_u = invertDic(model_clust2ent_u, 'm2os')
    true_ent2clust = invertDic(true_clust2ent, 'm2os')

    eval_results = evaluate(model_ent2clust_u, model_clust2ent_u, true_ent2clust, true_clust2ent)
    macro_prec, micro_prec, pair_prec = eval_results['macro_prec'], eval_results['micro_prec'], eval_results[
        'pair_prec']
    macro_recall, micro_recall, pair_recall = eval_results['macro_recall'], eval_results['micro_recall'], eval_results[
        'pair_recall']
    macro_f1, micro_f1, pair_f1 = eval_results['macro_f1'], eval_results['micro_f1'], eval_results['pair_f1']
    ave_prec = (macro_prec + micro_prec + pair_prec) / 3
    ave_recall = (macro_recall + micro_recall + pair_recall) / 3
    ave_f1 = (macro_f1 + micro_f1 + pair_f1) / 3
    model_clusters = len(model_clust2ent_u)
    model_Singletons = len([1 for _, clust in model_clust2ent_u.items() if len(clust) == 1])
    gold_clusters = len(true_clust2ent)
    gold_Singletons = len([1 for _, clust in true_clust2ent.items() if len(clust) == 1])
    if print_or_not:
        print('Ave-prec=', ave_prec, 'macro_prec=', macro_prec, 'micro_prec=', micro_prec,
              'pair_prec=', pair_prec)
        print('Ave-recall=', ave_recall, 'macro_recall=', macro_recall, 'micro_recall=', micro_recall,
              'pair_recall=', pair_recall)
        print('Ave-F1=', ave_f1, 'macro_f1=', macro_f1, 'micro_f1=', micro_f1, 'pair_f1=', pair_f1)
        print('Model: #Clusters: %d, #Singletons %d' % (model_clusters, model_Singletons))
        print('Gold: #Clusters: %d, #Singletons %d' % (gold_clusters, gold_Singletons))
        print()


class Find_Best_Result(object):
    """
    Learns embeddings for NPs and relation phrases
    """

    def __init__(self, params, side_info, logger, true_ent2clust, true_clust2ent, clean_ent_list, ent2embed, 
                 cluster_threshold_max, cluster_threshold_min):
        self.p = params
        self.logger = logger
        self.side_info = side_info
        self.true_ent2clust, self.true_clust2ent = true_ent2clust, true_clust2ent
        self.clean_ent_list = clean_ent_list
        self.ent2embed = ent2embed
        self.cluster_threshold_max, self.cluster_threshold_min = cluster_threshold_max, cluster_threshold_min

    def go(self):
        self.entity_view_embed = []
        for ent in self.clean_ent_list:
            id = self.side_info.ent2id[ent]
            if id in self.side_info.isSub:
                self.entity_view_embed.append(self.ent2embed[id])

        self.entity_linking_prior = dict()
        for id in self.side_info.ent_old_id2new_id:
            if id in self.side_info.isSub:
                self.entity_linking_prior.update({id: self.side_info.ent_old_id2new_id[id]})

        # print('clean_ent_list:', type(self.clean_ent_list), len(self.clean_ent_list))
        # print('self.entity_view_embed:', type(self.entity_view_embed), len(self.entity_view_embed))
        # print('self.entity_linking_prior:', type(self.entity_linking_prior), len(self.entity_linking_prior))  # 23735

        el_prior_cluster_list = []
        el_repeat_old_dict = dict()
        for i in range(len(self.clean_ent_list)):
            ent = self.clean_ent_list[i]
            old_id = self.side_info.ent2id[ent]
            if old_id in self.side_info.isSub:
                new_id = self.entity_linking_prior[old_id]
                el_prior_cluster_list.append(new_id)
                if new_id not in el_repeat_old_dict:
                    el_repeat_old_dict.update({new_id: 1})

        best_embed_threshold, best_el_embed_threshold = dict(), dict()
        best_embed_cluster_threshold, best_embed_ave_f1, best_el_embed_cluster_threshold, best_el_embed_ave_f1 = 0, 0, 0, 0

        for cluster_threshold in range(self.cluster_threshold_max, self.cluster_threshold_min, -1):
            self.cluster_threshold = cluster_threshold / 100
            # print('cluster_threshold:', self.cluster_threshold)
            np_view_clusters, np_view_clusters_center = HAC_getClusters(self.p, self.entity_view_embed,
                                                                        self.cluster_threshold)
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

            # ave_prec, ave_recall, ave_f1, macro_prec, micro_prec, pair_prec, macro_recall, micro_recall, \
            # pair_recall, macro_f1, micro_f1, pair_f1, model_clusters, model_Singletons, gold_clusters, gold_Singletons \
            #     = cluster_test(self.p, self.side_info, el_prior_cluster_list, self.true_ent2clust, self.true_clust2ent)
            # print('only use el prior dict ... ')
            # print('el_prior_cluster_list:', type(el_prior_cluster_list), len(el_prior_cluster_list),
            #       el_prior_cluster_list[0:10])
            # print('Ave-prec=', ave_prec, 'macro_prec=', macro_prec, 'micro_prec=', micro_prec,
            #       'pair_prec=', pair_prec)
            # print('Ave-recall=', ave_recall, 'macro_recall=', macro_recall, 'micro_recall=', micro_recall,
            #       'pair_recall=', pair_recall)
            # print('Ave-F1=', ave_f1, 'macro_f1=', macro_f1, 'micro_f1=', micro_f1, 'pair_f1=', pair_f1)
            # print('Model: #Clusters: %d, #Singletons %d' % (model_clusters, model_Singletons))
            # print('Gold: #Clusters: %d, #Singletons %d' % (gold_clusters, gold_Singletons))
            # print()

            ave_prec, ave_recall, ave_f1, macro_prec, micro_prec, pair_prec, macro_recall, micro_recall, \
            pair_recall, macro_f1, micro_f1, pair_f1, model_clusters, model_Singletons, gold_clusters, gold_Singletons \
                = cluster_test(self.p, self.side_info, np_view_clusters_list, self.true_ent2clust, self.true_clust2ent)
            # print('only use KG embedding ... ')
            # print('np_view_clusters_list:', type(np_view_clusters_list), len(np_view_clusters_list),
            #       np_view_clusters_list[0:10])
            # print('Ave-prec=', ave_prec, 'macro_prec=', macro_prec, 'micro_prec=', micro_prec,
            #       'pair_prec=', pair_prec)
            # print('Ave-recall=', ave_recall, 'macro_recall=', macro_recall, 'micro_recall=', micro_recall,
            #       'pair_recall=', pair_recall)
            # print('Ave-F1=', ave_f1, 'macro_f1=', macro_f1, 'micro_f1=', micro_f1, 'pair_f1=', pair_f1)
            # print('Model: #Clusters: %d, #Singletons %d' % (model_clusters, model_Singletons))
            # print('Gold: #Clusters: %d, #Singletons %d' % (gold_clusters, gold_Singletons))
            # print()
            best_embed_threshold.update({self.cluster_threshold: ave_f1})

            # crawl_el_cluster_list = []
            # for i in range(len(np_view_clusters_list)):
            #     old_id = np_view_clusters_list[i]
            #     new_id = el_prior_cluster_list[old_id]
            #     crawl_el_cluster_list.append(new_id)
            # print('crawl_el_cluster_list:', type(crawl_el_cluster_list), len(crawl_el_cluster_list))
            # ave_prec, ave_recall, ave_f1, macro_prec, micro_prec, pair_prec, macro_recall, micro_recall, \
            # pair_recall, macro_f1, micro_f1, pair_f1, model_clusters, model_Singletons, gold_clusters, gold_Singletons \
            #     = cluster_test(self.p, self.side_info, crawl_el_cluster_list, self.true_ent2clust, self.true_clust2ent)
            # print('crawl + el ... ')
            # print('Ave-prec=', ave_prec, 'macro_prec=', macro_prec, 'micro_prec=', micro_prec,
            #       'pair_prec=', pair_prec)
            # print('Ave-recall=', ave_recall, 'macro_recall=', macro_recall, 'micro_recall=', micro_recall,
            #       'pair_recall=', pair_recall)
            # print('Ave-F1=', ave_f1, 'macro_f1=', macro_f1, 'micro_f1=', micro_f1, 'pair_f1=', pair_f1)
            # print('Model: #Clusters: %d, #Singletons %d' % (model_clusters, model_Singletons))
            # print('Gold: #Clusters: %d, #Singletons %d' % (gold_clusters, gold_Singletons))
            # print()

            el_crawl_cluster_list = []
            for i in range(len(el_prior_cluster_list)):
                old_id = el_prior_cluster_list[i]
                new_id = np_view_clusters_list[old_id]
                el_crawl_cluster_list.append(new_id)
            # print('el_crawl_cluster_list:', type(el_crawl_cluster_list), len(el_crawl_cluster_list))
            ave_prec, ave_recall, ave_f1, macro_prec, micro_prec, pair_prec, macro_recall, micro_recall, \
            pair_recall, macro_f1, micro_f1, pair_f1, model_clusters, model_Singletons, gold_clusters, gold_Singletons \
                = cluster_test(self.p, self.side_info, el_crawl_cluster_list, self.true_ent2clust, self.true_clust2ent)
            # print('EL dict + KG embedding ... ')
            # print('Ave-prec=', ave_prec, 'macro_prec=', macro_prec, 'micro_prec=', micro_prec,
            #       'pair_prec=', pair_prec)
            # print('Ave-recall=', ave_recall, 'macro_recall=', macro_recall, 'micro_recall=', micro_recall,
            #       'pair_recall=', pair_recall)
            # print('Ave-F1=', ave_f1, 'macro_f1=', macro_f1, 'micro_f1=', micro_f1, 'pair_f1=', pair_f1)
            # print('Model: #Clusters: %d, #Singletons %d' % (model_clusters, model_Singletons))
            # print('Gold: #Clusters: %d, #Singletons %d' % (gold_clusters, gold_Singletons))
            # print()
            best_el_embed_threshold.update({self.cluster_threshold: ave_f1})

        for cluster_threshold in range(self.cluster_threshold_max, self.cluster_threshold_min, -1):
            cluster_threshold_real = cluster_threshold / 100
            value = best_embed_threshold[cluster_threshold_real]
            if value > best_embed_ave_f1:
                best_embed_cluster_threshold = cluster_threshold_real
                best_embed_ave_f1 = value
            else:
                continue
        print(best_embed_threshold)
        print('best_embed_cluster_threshold:', best_embed_cluster_threshold, 'best_embed_ave_f1:', best_embed_ave_f1)
        np_view_clusters, np_view_clusters_center = HAC_getClusters(self.p, self.entity_view_embed,
                                                                    best_embed_cluster_threshold)
        np_view_clusters = list(np_view_clusters)
        best_np_view_clusters_list = []
        el_repeat_dict = dict()
        np_result_dict = dict()
        num = 0
        for i in range(len(np_view_clusters)):
            np_id = np_view_clusters[i]
            el_id = el_prior_cluster_list[i]
            if np_id in np_result_dict:
                best_np_view_clusters_list.append(np_result_dict[np_id])
            else:
                if el_id in el_repeat_dict:
                    while num in best_np_view_clusters_list or num in el_repeat_old_dict:
                        num += 1
                    np_result_dict.update({np_id: num})
                    best_np_view_clusters_list.append(np_result_dict[np_id])
                    num += 1
                else:
                    best_np_view_clusters_list.append(el_id)
                    np_result_dict.update({np_id: el_id})
                    el_repeat_dict.update({el_id: 1})
            if el_id not in el_repeat_dict:
                el_repeat_dict.update({el_id: 1})

        ave_prec, ave_recall, ave_f1, macro_prec, micro_prec, pair_prec, macro_recall, micro_recall, \
        pair_recall, macro_f1, micro_f1, pair_f1, model_clusters, model_Singletons, gold_clusters, gold_Singletons \
            = cluster_test(self.p, self.side_info, best_np_view_clusters_list, self.true_ent2clust, self.true_clust2ent)
        print('only use KG embedding ... ')
        # print('np_view_clusters_list:', type(np_view_clusters_list), len(np_view_clusters_list),
        #       np_view_clusters_list[0:10])
        print('Ave-prec=', ave_prec, 'macro_prec=', macro_prec, 'micro_prec=', micro_prec,
              'pair_prec=', pair_prec)
        print('Ave-recall=', ave_recall, 'macro_recall=', macro_recall, 'micro_recall=', micro_recall,
              'pair_recall=', pair_recall)
        print('Ave-F1=', ave_f1, 'macro_f1=', macro_f1, 'micro_f1=', micro_f1, 'pair_f1=', pair_f1)
        print('Model: #Clusters: %d, #Singletons %d' % (model_clusters, model_Singletons))
        print('Gold: #Clusters: %d, #Singletons %d' % (gold_clusters, gold_Singletons))
        print()

        for cluster_threshold in range(self.cluster_threshold_max, self.cluster_threshold_min, -1):
            cluster_threshold_real = cluster_threshold / 100
            value = best_el_embed_threshold[cluster_threshold_real]
            if value > best_el_embed_ave_f1:
                best_el_embed_cluster_threshold = cluster_threshold_real
                best_el_embed_ave_f1 = value
            else:
                continue
        print('best_el_embed_cluster_threshold:', best_el_embed_cluster_threshold, 'best_el_embed_ave_f1:',
              best_el_embed_ave_f1)
        np_view_clusters, np_view_clusters_center = HAC_getClusters(self.p, self.entity_view_embed,
                                                                    best_el_embed_cluster_threshold)
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

        best_el_crawl_cluster_list = []
        for i in range(len(el_prior_cluster_list)):
            old_id = el_prior_cluster_list[i]
            new_id = np_view_clusters_list[old_id]
            best_el_crawl_cluster_list.append(new_id)
        # print('el_crawl_cluster_list:', type(el_crawl_cluster_list), len(el_crawl_cluster_list))
        ave_prec, ave_recall, ave_f1, macro_prec, micro_prec, pair_prec, macro_recall, micro_recall, \
        pair_recall, macro_f1, micro_f1, pair_f1, model_clusters, model_Singletons, gold_clusters, gold_Singletons \
            = cluster_test(self.p, self.side_info, best_el_crawl_cluster_list, self.true_ent2clust, self.true_clust2ent, print_or_not=True, get_cluster_print=False)
        print('EL dict + KG embedding ... ')
        print('Ave-prec=', ave_prec, 'macro_prec=', macro_prec, 'micro_prec=', micro_prec,
              'pair_prec=', pair_prec)
        print('Ave-recall=', ave_recall, 'macro_recall=', macro_recall, 'micro_recall=', micro_recall,
              'pair_recall=', pair_recall)
        print('Ave-F1=', ave_f1, 'macro_f1=', macro_f1, 'micro_f1=', micro_f1, 'pair_f1=', pair_f1)
        print('Model: #Clusters: %d, #Singletons %d' % (model_clusters, model_Singletons))
        print('Gold: #Clusters: %d, #Singletons %d' % (gold_clusters, gold_Singletons))
        print()

        return best_embed_threshold, best_np_view_clusters_list, best_el_crawl_cluster_list, el_prior_cluster_list


