import clusters

docs,words,data=clusters.readfile('titles_vectors.txt')
rdata=clusters.rotatematrix(data)
num_clusters = 2
print ('Grouping words into {} clusters:'.format(num_clusters))

print()
clust=clusters.kcluster(rdata,distance=clusters.pearson,k=num_clusters)
print ('clusters by pearson correlation')
for i in range(num_clusters):
    print("cluster {}".format(i+1))
    print ([words[r-1] for r in clust[i]])

print()
clust=clusters.kcluster(rdata,distance=clusters.tanimoto,k=num_clusters)
print ('clusters by tanimoto coefficient')
for i in range(num_clusters):
    print("cluster {}".format(i+1))
    print ([words[r] for r in clust[i]])

print()
clust=clusters.kcluster(rdata,distance=clusters.euclidean,k=num_clusters)
print ('clusters by euclidean distance')
for i in range(num_clusters):
    print("cluster {}".format(i+1))
    print ([words[r] for r in clust[i]])

print()
clust=clusters.kcluster(rdata,distance=clusters.cosine,k=num_clusters)
print ('clusters by cosine distance')
for i in range(num_clusters):
    print("cluster {}".format(i+1))
    print ([words[r] for r in clust[i]])
