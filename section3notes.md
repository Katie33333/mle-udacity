## Unsupervised Learning

**Unsupervised Learning** - to find structure in the data without labels

### Clustering

Group data into different classes

**K Means** - this algorithm is a clustering method that moves centroids (center points of classes) and reassign points to the nearest centroid

1. Assign - which points are closest to the center of each class

2. Optimize - minimize the total credand? distance of the point to the center of the class - minimize the total quadratic length

Challenges - determining how many clusters you want

Limitations - K Means is called the hill climbing algorithm and as a result, it's very dependant on where you put your cluster centers.  The output of any fixed training set (and number of cluster centers) will not always be the same. 

K Means clustering in sklearn - n_iters = how many times does it come up with clusters, max_iter, n_clusters - determines how many clusters you want

### Dimensionality Reduction - example of taking multiple points (2dimensional space) and making a line through them (reduced to a line)
