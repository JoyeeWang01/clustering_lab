import clusters

docs,words,data=clusters.readfile('titlesdata.txt')
rdata=clusters.rotatematrix(data)

clust=clusters.hcluster(rdata,distance=clusters.pearson)
print('clusters by pearson correlation')
clusters.printclust(clust,labels=words)
clusters.drawdendrogram(clust,words,jpeg='wordsclustpearson.jpg')

clust=clusters.hcluster(rdata,distance=clusters.tanimoto)
print('clusters by tanimoto coefficient')
clusters.printclust(clust,labels=words)
clusters.drawdendrogram(clust,words,jpeg='wordsclusttanimoto.jpg')

clust=clusters.hcluster(rdata,distance=clusters.euclidean)
print('clusters by euclidean distance')
clusters.printclust(clust,labels=words)
clusters.drawdendrogram(clust,words,jpeg='wordsclusteuclidean.jpg')
