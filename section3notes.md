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

### Clustering Methods beyond K Means
K Means is good when you have data that looks clustered and you have a good ideas of how many clusters
Can't always find the right grouping since we rely on distance to center (Think of the 2 rings example dataset)

### Hierarchical Clustering
**Single link clustering** - find distances between points and then distances between clusters - draw the arch picture called a dendrogram - can specify the number of clusters.  Cons - can have oblong shaped clusters or one cluster that eats up most of the dataset.  Even if it can't cluster correctly, you can maybe find insights in the dendrogram.  Can't do single link in scikit learn
**Complete Link** - starts off the same as single link, but instead of looking for the smallest distance between a point and a cluster, it looks for the distances between the point and the 2 furthest points between 2 clusters. - Produces compact clusters.  Con - only looks at the one furthest point and disregards others 
**Average Link** - looks at the distance between a point and the average distance for all other points
**Ward's method** - default method in scikit learn and looks to minimize variance between clusters - finds a central point in two clusters, takes an average for all other distances in the cluster, to the power of 2

**Implementation** - (See Lab in projects folder) scikit learn for ward's method and scipy for drawing dendrograms (see screen shots)
To determine which clustering result better matches the original labels of the samples, we can use **adjusted_rand_score** which is an external cluster validation index which results in a score between -1 and 1, where 1 means two clusterings are identical of how they grouped the samples in a dataset (regardless of what label is assigned to each cluster).

Looking at this, we can see that the forth column has smaller values than the rest of the columns, and so its variance counts for less in the clustering process (since clustering is based on distance). Let us **normalize** the dataset so that each dimension lies between 0 and 1, so they have equal weight in the clustering process.
This is done by subtracting the minimum from each column then dividing the difference by the range.
sklearn provides us with a useful utility called preprocessing.normalize() that can do that for us

Pros of HC: hierarchical representation can be very informative, visualization abilities
Cons of HC: sensitive to noise and outliers

**Density Based Spatial Clustering of applications with noise** - DBSCAN - density based clustering with datasets with noise, do not have to specify a number of clusters - labels data points that are clustered together and other outliers as noise
* Core points
* Noise points - does not have at least 5 points in its cluster (or whatever you set min_samples to)
* Boarder point - part of cluster but not a core point
