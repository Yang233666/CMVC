'''
Performs clustering operation on learned embeddings for both NP and relations
Uses HAC method for clustering.
'''

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


class Clustering(object):
	def __init__(self, ent2embed, rel2embed, side_info, params, cluster_threshold_real, issub=True):
		self.p = params
		self.side_info = side_info
		self.ent2embed = ent2embed
		self.rel2embed = rel2embed
		self.cluster_threshold_real = cluster_threshold_real
		self.issub = issub

		raw_ent_clust 	= self.getClusters(self.ent2embed) 			 # Clustering entities
		self.ent_clust	= self.getEntRep(raw_ent_clust, self.side_info.ent_freq)  # Finding entity cluster representative

		#raw_rel_clust 	= self.getClusters(self.rel2embed)			 # Clustering relations
		#self.rel_clust	= self.getRelRep(raw_rel_clust)				 # Finding relation cluster representative

	def getClusters(self, embed):

		n, m 	= len(embed), self.p.embed_dims
		X 	= np.empty((n, m), np.float32)

		for i in range(len(embed)):
			X[i, :] = embed[i]

		dist 	  = pdist(X, 	  metric=self.p.metric)
		clust_res = linkage(dist, method=self.p.linkage)
		labels    = fcluster(clust_res, t=self.cluster_threshold_real, criterion='distance') - 1
		clusters  = [[] for i in range(max(labels) + 1)]

		for i in range(len(labels)):
			clusters[labels[i]].append(i)
		return clusters

	def getEntRep(self, clusters, ent2freq):
		final_res = dict()
		if self.issub:
			for cluster in clusters:
				rep, max_freq = cluster[0], -1
				for ent in cluster:
					if ent2freq[ent] > max_freq:
						max_freq, rep = ent2freq[ent], ent
				rep = self.side_info.id2sub[rep]
				cluster = [self.side_info.id2sub[ele] for ele in cluster]
				final_res[rep] = cluster
		else:
			for cluster in clusters:
				rep, max_freq = cluster[0], -1
				for ent in cluster:
					if ent2freq[ent] > max_freq:
						max_freq, rep = ent2freq[ent], ent
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