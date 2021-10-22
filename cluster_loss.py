'''
Performs clustering operation on learned embeddings for both NP and relations
Uses HAC method for clustering.
'''
from helper import *

from joblib import Parallel, delayed
import numpy as np, time, random, pdb, itertools

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

from kmeans import *


class Clustering(object):
	def __init__(self, ent2embed, rel2embed, params, cluster_threshold_real, if_average_loss=True):
		self.p = params
		ent2embed = ent2embed.squeeze(1)
		rel2embed = rel2embed.squeeze(1)
		self.cluster_threshold_real = cluster_threshold_real
		self.if_average_loss = if_average_loss

		num_clusters = 6600
		kmeans_train(dataset=ent2embed, num_centers=num_clusters)
		exit()

		# self.ent_clust, self.ent_loss = self.getClusters(self.ent2embed, if_average_loss=self.if_average_loss)  # Clustering entities


	def getClusters(self, embed, if_average_loss=True):
		d_in = 0
		dist 	  = pdist(embed, 	  metric=self.p.metric)
		clust_res = linkage(dist, method=self.p.linkage)
		if if_average_loss:
			d_out = clust_res[:, 2].sum() / len(clust_res[:, 2])
		else:
			d_out = clust_res[:, 2].sum()
		labels    = fcluster(clust_res, t=self.cluster_threshold_real, criterion='distance') - 1
		clusters  = [[] for i in range(max(labels) + 1)]

		for i in range(len(labels)): 
			clusters[labels[i]].append(i)

		num = 0
		for cluster in clusters:
			if len(cluster) > 1:
				num += 1
				d_in += self.clusters_inner_distance(self.ent2embed, cluster)

		if if_average_loss:
			d_in = d_in / num

		loss = self.max_margin_loss(d_in, d_out, self.p.single_gamma)
		return clusters, loss

	def clusters_inner_distance(self, embed, cluster_list):
		n, m = len(cluster_list), self.p.embed_dims
		X = np.empty((n, m), np.float32)

		for i in range(len(cluster_list)):
			cluster_id = cluster_list[i]
			X[i, :] = embed[cluster_id]

		dist = pdist(X, metric=self.p.metric)
		return dist.sum()

	def max_margin_loss(self, positive_score, negative_score, gamma):
		score = positive_score - negative_score + gamma
		loss = max(score, 0)
		return loss
