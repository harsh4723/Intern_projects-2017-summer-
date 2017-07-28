"""
DBSCAN test on multilabel data

"""


import numpy as np
import csv
import math
from sklearn.cluster import DBSCAN
from sklearn import metrics




loc=6
def dataload(filename):													#load data and convert into a list
	with open(filename,'rb') as csvfile:
		dataset=csv.reader(csvfile)
		data=list(dataset)
		data.remove([])
		datafinal=[]
		c=0
		for x in data:
			if(x==[]):
				c=0
			elif(x[0]=='@data'):
				c=1
			if(c==1):
				datafinal.append(x)
		del(datafinal[0])
		for x in range(len(datafinal)):
			for y in range(len(datafinal[0])):
				datafinal[x][y]=float(datafinal[x][y])

	
		return datafinal

def eucliddistance(x1,x2):
	sum=0
	for i in range(len(x1)):
		d1=pow((x1[i]-x2[i]),2)
		sum=sum+d1
	return math.sqrt(sum)


data=dataload('scene-train.csv')
lod=len(data[0])
X=[]
for i in range(len(data)):
	X.append(data[i][:(lod-loc)])

# X is sample data
l= len(X) 

distmatrix=[]
for i in range(l):
	li=[]
	for j in range(l):
		li.append(eucliddistance(X[i],X[j]))
	distmatrix.append(li)

mins=[]
for x in distmatrix:
	x.remove(0.0)
	mins.append(min(x))

mins = [x for x in mins if x != 0.0]
print min(mins)
d=sum(mins)/float(len(mins)) #taking epsilon as average


# Compute DBSCAN
db = DBSCAN(eps=d, min_samples=3).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
print labels
# Number of clusters in labels, ignoring noise if present.
print len(labels)
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print "no of clusters " +str(n_clusters_)
