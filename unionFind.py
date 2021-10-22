"""UnionFind.py

Union-find data structure. Based on Josiah Carlson's code,
http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/215912
with significant additional changes by D. Eppstein.
"""
import pdb

class DisjointSet(object):

	def __init__(self):
		self.leader = {} # maps a member to the group's leader
		self.group = {} # maps a group leader to the group (which is a set)

	def add(self, a, b):
		leadera = self.leader.get(a)
		leaderb = self.leader.get(b)
		if leadera is not None:
			if leaderb is not None:
				if leadera == leaderb: return # nothing to do
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