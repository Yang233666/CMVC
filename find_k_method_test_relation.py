import numpy as np
import pickle
from sklearn import datasets  # 导入库
from tqdm import tqdm
from find_k_method import Inverse_JumpsMethod
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# dataset = 'OPIEC'
# dataset = 'reverb45k_change'
dataset = 'NYTimes2018'

choice = 'crawl'
# choice = 'relation'
# choice = 'BERT'

n_jobs=10
# n_jobs=1

print('dataset:', dataset)

if dataset == 'OPIEC' or dataset == 'reverb45k_change' or dataset == 'NYTimes2018':
    if choice == 'crawl':
        # fname1 = '../file/' + dataset + '/1E_init'
        fname1 = '../file/' + dataset + '/1R_init'
    elif choice == 'BERT':
        if dataset == 'OPIEC':
            fname1 = '../file/' + dataset + '/' + 'multi_view/context_view_1.2/first' + '/bert_cls_el_0_19'
        else:
            fname1 = '../file/' + dataset + '/' + 'multi_view/context_view_1.2/first' + '/bert_cls_el_0_0'
    else:
        fname1 = '../file/' + dataset + '/multi_view/relation_view_1.2_50000_1000_2000/entity_embedding'
    print('choice:', choice)
    print('fname1:', fname1)
    E_init = pickle.load(open(fname1, 'rb'))
    print('E_init:', type(E_init), E_init.shape)
    input_embed = E_init
    if dataset == 'OPIEC':
        cluster_num = 490
    else:
        cluster_num = 6700
else:
    if dataset == 'digits':
        input_data, labels = datasets.load_digits(return_X_y=True)
    elif dataset == 'iris':
        input_data, labels = datasets.load_iris(return_X_y=True)
    else:
        input_data, labels = datasets.load_wine(return_X_y=True)
    (n_samples, n_features), n_digits = input_data.shape, np.unique(labels).size
    print(
        f"# digits: {n_digits}; # samples: {n_samples}; # features {n_features}"
    )
    print('input_data:', type(input_data), input_data.shape)
    print('labels:', type(labels), labels.shape)
    cluster_num = len(list(set(labels.tolist())))
    print('cluster_num:', cluster_num)
    input_embed = input_data
print('input_embed:', type(input_embed), input_embed.shape)

if dataset == 'OPIEC':
    level_one_min, level_one_max, level_one_gap = 200, 1200, 100
    # level_one_min, level_one_max, level_one_gap = 4, 6, 1
elif dataset == 'reverb45k_change':
    # level_one_min, level_one_max, level_one_gap = 4000, 10000, 1000
    level_one_min, level_one_max, level_one_gap = 2000, 10000, 1000
elif dataset == 'NYTimes2018':
    # level_one_min, level_one_max, level_one_gap = 2000, 7000, 1000  # level ont
    level_one_min, level_one_max, level_one_gap = 100, 2000, 100  # level two
# elif dataset == 'digits':
#     level_one_min, level_one_max, level_one_gap = 1, 15, 1
# elif dataset == 'iris':
#     level_one_min, level_one_max, level_one_gap = 1, 5, 1
else:
    level_one_min, level_one_max, level_one_gap = 1, 13, 1
cluster_list = range(level_one_min, level_one_max, level_one_gap)
print('第一层， 聚类粒度为:', level_one_gap)
print('level_one_min, level_one_max, level_one_gap:', level_one_min, level_one_max, level_one_gap)
k_list = list(cluster_list)

if choice == 'BERT':
    dim_is_bert = True
else:
    dim_is_bert = False

method2first_cluster_num_dict = dict()
jm = Inverse_JumpsMethod(data=input_embed, k_list=cluster_list, dim_is_bert=dim_is_bert)
# print('number of dimensions:', jm.p)
jm.Distortions(random_state=0)
distortions = jm.distortions
jm.Jumps(distortions=distortions)
jumps = jm.jumps
# print('jumps:', type(jumps), len(jumps), jumps)
level_one_Inverse_JumpsMethod = jm.recommended_cluster_number
# print('new jump level_one_k:', level_one_k)

# level_one_Inverse_JumpsMethod = 5000
print('Inverse_JumpsMethod k:', level_one_Inverse_JumpsMethod)
method2first_cluster_num_dict['Inverse_JumpsMethod'] = level_one_Inverse_JumpsMethod
# opiec crawl hac k=1310
# opiec bert kmeans k=1000

print('Inverse_JumpsMethod k:', level_one_Inverse_JumpsMethod)
exit()

if dataset == 'OPIEC' or dataset == 'reverb45k_change' or dataset == 'NYTimes2018':
    print('Level two:')
    data = input_embed

    for method in method2first_cluster_num_dict:
        level_one_k = method2first_cluster_num_dict[method]
        print('Method:', method, 'level_one_k:', level_one_k)
        level_two_min, level_two_max, level_two_gap = level_one_k - level_one_gap, level_one_k + level_one_gap, int(
            level_one_gap / 10)
        minK, maxK = level_two_min, level_two_max
        cluster_list = range(level_two_min, level_two_max, level_two_gap)
        k_list = list(cluster_list)
        est_k = 0
        print('level_two_min, level_two_max, level_two_gap:', level_two_min, level_two_max, level_two_gap)

        if method == 'Inverse_JumpsMethod':
            jm = Inverse_JumpsMethod(data=input_embed, k_list=cluster_list, dim_is_bert=dim_is_bert)
            # print('number of dimensions:', jm.p)
            jm.Distortions(random_state=0)
            distortions = jm.distortions
            jm.Jumps(distortions=distortions)
            jumps = jm.jumps
            # print('jumps:', type(jumps), len(jumps), jumps)
            est_k = jm.recommended_cluster_number

        print(method, ' k: ', est_k)
        print()

    print('Golden cluster number : ', cluster_num)
    print()