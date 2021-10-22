import numpy as np
import pickle
from find_k_method import JumpsMethod

dataset = 'OPIEC'
# dataset = 'reverb45k_change'
choice = 'crawl'
# choice = 'relation'
# choice = 'BERT'
method = 'kmeans'
# method = 'hac'

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

jm = JumpsMethod(data=subject_embed, cluster_list=cluster_list, dataset=dataset, dim_is_bert=dim_is_bert)
print('number of dimensions:', jm.p)
# jm.Distortions(random_state=0)
# distortions = jm.distortions
distortions = [0.00082123, 0.00066695, 0.00056035, 0.00046376, 0.00038816, 0.00033198,
 0.00027615, 0.00022663, 0.00019042, 0.00015824, 0.00013134]
print('distortions:', type(distortions), len(distortions), distortions)
jm.Jumps(distortions=distortions)
jumps = jm.jumps
print('jumps:', type(jumps), len(jumps), jumps)
level_one_k = jm.recommended_cluster_number
print('level_one_k:', level_one_k)
# opiec crawl kmeans k=500
# opiec relation kmeans k=500
# opiec bert kmeans k=900

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
distortions = [0.00056035, 0.00055435, 0.00054289, 0.00053378, 0.00052616, 0.00050889,
 0.0004997,  0.00048848, 0.0004808,  0.00047419, 0.00046376, 0.00045941,
 0.0004488,  0.00044178, 0.00043364, 0.00042268, 0.00041709, 0.0004102,
 0.00040302, 0.00039467, 0.00038816]
print('distortions:', type(distortions), len(distortions), distortions)
jm.Jumps(distortions=distortions)
jumps = jm.jumps
print('jumps:', type(jumps), len(jumps), jumps)
k = jm.recommended_cluster_number
print('k:', k)

# opiec crawl kmeans k = 490
# opiec relation kmeans k=540
# opiec bert kmeans k = 970