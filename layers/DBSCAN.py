import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# #############################################################################
def getDBSCAN(points, eps=0.135, min_samples=2,plot=True):
    X = StandardScaler().fit_transform(points)
    X[:,0] *= 5
    # #############################################################################
    # Compute DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("number of clusters", n_clusters_)

    if plot:
        # Black removed and is used for noise instead.
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = (labels == k)

            xy = X[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], -xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=4)

            xy = X[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], -xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=2)

        plt.title('Estimated number of clusters: %d' % n_clusters_)
    return labels,n_clusters_

### sort points by labels
def getClusters(X,labels,n,plot=False):
    clusters = []
    for i in range(0,n):
        cluster = np.array(X)[labels==i]
        c =sorted(cluster, key=lambda z: z[1])
        c = np.array(c)
        clusters.append(c)
        if plot:
            plt.plot(c[:,0],c[:,1])
    final = sorted(clusters, key=lambda z: z[:,0].sum()/len(z))
    if plot:
        plt.show()
    return np.array(final)
        

##print('Estimated number of clusters: %d' % n_clusters_)
##print('Estimated number of noise points: %d' % n_noise_)
##print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
##print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
##print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
##print("Adjusted Rand Index: %0.3f"
##      % metrics.adjusted_rand_score(labels_true, labels))
##print("Adjusted Mutual Information: %0.3f"
##      % metrics.adjusted_mutual_info_score(labels_true, labels))
##print("Silhouette Coefficient: %0.3f"
##      % metrics.silhouette_score(X, labels))



