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
	def __init__(self, ent2embed, ent_freq, id2sub, params, cluster_threshold_real):
		self.p = params
		self.ent_freq = ent_freq
		self.ent2embed = ent2embed
		self.cluster_threshold_real = cluster_threshold_real
		self.id2sub = id2sub

		raw_ent_clust 	= self.getClusters(self.ent2embed) 			 # Clustering entities
		self.ent_clust	= self.getEntRep(raw_ent_clust, self.ent_freq)  # Finding entity cluster representative


	def getClusters(self, embed):

		n, m 	= len(embed), self.p.embed_dims
		X 	= np.empty((n, m), np.float32)

		for i in range(len(embed)): 
			X[i, :] = embed[i]

		dist 	  = pdist(X, 	  metric=self.p.metric)
		clust_res = linkage(dist, method=self.p.linkage)
		labels    = fcluster(clust_res, t=self.cluster_threshold_real, criterion='distance') - 1
		#labels = fcluster(clust_res, t=self.p.thresh_val, criterion='distance') - 1
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

			rep     = self.id2sub[rep]
			cluster = [self.id2sub[ele] for ele in cluster]

			final_res[rep] = cluster

		return final_res