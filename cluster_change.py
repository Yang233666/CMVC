'''
Performs clustering operation on learned embeddings for both NP and relations
Uses HAC method for clustering.
'''
from helper import *

from joblib import Parallel, delayed
import numpy as np, time, random, pdb, itertools

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

from metrics import *

class Clustering(object):
	def __init__(self, ent2embed, rel2embed, side_info, params, sub2name_embed, rel2name_embed, cluster_threshold_real, weight_real):
		self.p = params
		self.side_info = side_info
		self.ent2embed = ent2embed
		self.rel2embed = rel2embed
		self.sub2name_embed = sub2name_embed
		self.rel2name_embed = rel2name_embed

		raw_ent_clust 	= self.getClusters(self.ent2embed, self.sub2name_embed, cluster_threshold_real, weight_real) 			 # Clustering entities
		self.ent_clust	= self.getEntRep(raw_ent_clust, self.side_info.ent_freq) # Finding entity cluster representative

		#raw_rel_clust 	= self.getClusters(self.rel2embed, self.rel2name_embed, cluster_threshold_real, weight_real)			 # Clustering relations
		#self.rel_clust	= self.getRelRep(raw_rel_clust)				 # Finding relation cluster representative

	def getClusters(self, embed, name_embed, cluster_threshold_real, weight_real):
		print('cluster_threshold:', cluster_threshold_real)
		print('weight_real:', weight_real, (1-weight_real))

		n, m 	= len(embed), self.p.embed_dims
		X 	= np.empty((n, m), np.float32)
		X_name = np.empty((n, m), np.float32)

		for i in range(len(embed)): 
			X[i, :] = embed[i]
			X_name[i, :] = name_embed[i]

		dist 	  = pdist(X, 	  metric=self.p.metric)
		dist_name = pdist(X_name, metric=self.p.metric)

		for i in range(len(dist_name)):
			if np.isfinite(dist_name[i]):continue
			else:dist_name[i] = dist[i]

		dist_ave = np.float64(weight_real * dist + (1-weight_real) * dist_name)

		'''
		for i in range(len(dist_ave)):
			if np.isfinite(dist_ave[i]):continue
			else:dist_ave[i] = dist[i]
		'''

		clust_res = linkage(dist_ave, method=self.p.linkage)
		labels    = fcluster(clust_res, t=cluster_threshold_real, criterion='distance') - 1
		clusters  = [[] for i in range(max(labels) + 1)]

		for i in range(len(labels)): 
			clusters[labels[i]].append(i)

		return clusters

	def getEntRep(self, clusters, ent2freq):
		final_res = dict()

		for cluster in clusters:
			rep, max_freq = cluster[0], -1

			for ent in cluster:
				if ent2freq[ent] > max_freq:
					max_freq, rep = ent2freq[ent], ent

			rep     = self.side_info.id2sub[rep]
			cluster = [self.side_info.id2sub[ele] for ele in cluster]

			final_res[rep] = cluster

		return final_res

	def getRelRep(self, clusters):
		embed 	  = self.rel2embed
		final_res = {}

		for cluster in clusters:
			# Find the centroid vector for the elements in cluster
			centroid = np.zeros(self.p.embed_dims)
			for phr in cluster: centroid += embed[phr]
			centroid = centroid / len(cluster)

			# Find word closest to the centroid
			min_dist = float('inf')
			for rel in cluster:
				dist = np.linalg.norm(centroid - embed[rel])

				if dist < min_dist:
					min_dist = dist
					rep = rel

			final_res[rep] = cluster

		return final_res