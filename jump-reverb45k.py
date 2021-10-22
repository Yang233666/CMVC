import numpy as np
import pickle
from find_k_method import JumpsMethod, JumpsMethod_orgin

# dataset = 'OPIEC'
dataset = 'reverb45k_change'
choice = 'crawl'
# choice = 'relation'
# choice = 'BERT'

if choice == 'crawl':
    fname1 = '../file/' + dataset + '/1E_init'
elif choice == 'BERT':
    if dataset == 'OPIEC':
        fname1 = '../file/' + dataset + '/' + 'multi_view/context_view_1.2/first' + '/bert_cls_el_0_19'
    else:
        fname1 = '../file/' + dataset + '/' + 'multi_view/context_view_1.2/first' + '/bert_cls_el_0_0'
else:
    fname1 = '../file/' + dataset + '/multi_view/relation_view_1.2_50000_1000_2000/entity_embedding'
fname2, fname3 =  '../file/' + dataset + '/self.ent2id', '../file/' + dataset + '/self.isSub'
print('choice:', choice)
print('fname1:', fname1)
E_init = pickle.load(open(fname1, 'rb'))
ent2id = pickle.load(open(fname2, 'rb'))
isSub = pickle.load(open(fname3, 'rb'))
print('E_init:', type(E_init), E_init.shape)
print('ent2id:', type(ent2id), len(ent2id))
print('isSub:', type(isSub), len(isSub))
subject_embed = []
for ent in ent2id:
    id = ent2id[ent]
    if id in isSub:
        subject_embed.append(E_init[id])
subject_embed = np.array(subject_embed)
print('subject_embed:', type(subject_embed), subject_embed.shape)

print('dataset:', dataset)
method = 'kmeans'
# method = 'hac'
print('method:', method)
if method == 'hac':
    level_one_min, level_one_max, level_one_gap = 0.9, 0.2, -0.1
    cluster_list = np.arange(level_one_min, level_one_max, level_one_gap)
else:
    if dataset == 'OPIEC':
        level_one_min, level_one_max, level_one_gap = 200, 1200, 100
    else:
        level_one_min, level_one_max, level_one_gap = 4000, 10000, 1000
    cluster_list = range(level_one_min, level_one_max, level_one_gap)
print('第一层， 聚类粒度为:', level_one_gap)
print('level_one_min, level_one_max, level_one_gap:', level_one_min, level_one_max, level_one_gap)

if choice == 'BERT':
    dim_is_bert = True
else:
    dim_is_bert = False

# jm = JumpsMethod_orgin(data=subject_embed)
jm = JumpsMethod(data=subject_embed, cluster_list=cluster_list, dataset=dataset, dim_is_bert=dim_is_bert)
print('number of dimensions:', jm.p)
# jm.Distortions(random_state=0)
# distortions = jm.distortions
distortions = [4.1920951e-04, 3.1142912e-04, 2.2376596e-04, 1.6609141e-04, 1.2458483e-04,
 8.7930028e-05, 5.4903936e-05]  # # reverb45k_change crawl kmeans k=6000
print('distortions:', type(distortions), len(distortions), distortions)
jm.Jumps(distortions=distortions)
jumps = jm.jumps
print('jumps:', type(jumps), len(jumps), jumps)
level_one_k = jm.recommended_cluster_number
print('level_one_k:', level_one_k)
# reverb45k_change crawl kmeans k=6000
# reverb45k_change relation kmeans k=3000
# reverb45k_change bert kmeans k=3000

level_two_min, level_two_max, level_two_gap = level_one_k - level_one_gap, level_one_k + level_one_gap, int(level_one_gap / 10)
print('第二层， 聚类粒度为:', level_two_gap)
print('level_two_min, level_two_max, level_two_gap:', level_two_min, level_two_max, level_two_gap)
if method == 'hac':
    cluster_list = np.arange(level_two_min, level_two_max, level_two_gap)
else:
    cluster_list = range(level_two_min, level_two_max, level_two_gap)
jm = JumpsMethod(data=subject_embed, cluster_list=cluster_list, dataset=dataset, dim_is_bert=dim_is_bert)
print('number of dimensions:', jm.p)
# jm.Distortions(random_state=0)
# distortions = jm.distortions
distortions = [0.00031143, 0.00030198, 0.00029166, 0.00028205, 0.0002728,  0.00026358,
 0.00025556, 0.00024754, 0.00023878, 0.00023223, 0.00022377, 0.0002167,
 0.00021048, 0.00020326, 0.00019709, 0.00019115, 0.00018546, 0.00018019,
 0.00017509, 0.00017062, 0.00016609]  # # reverb45k_change crawl kmeans k=6700
print('distortions:', type(distortions), len(distortions), distortions)
jm.Jumps(distortions=distortions)
jumps = jm.jumps
print('jumps:', type(jumps), len(jumps), jumps)
k = jm.recommended_cluster_number
print('k:', k)
# reverb45k_change crawl kmeans k=6700
# reverb45k_change relation kmeans k=3000
# reverb45k_change bert kmeans k=3300