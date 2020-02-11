#/*==========================================================================================*\
#**                        _           _ _   _     _  _         _                            **
#**                       | |__  _   _/ | |_| |__ | || |  _ __ | |__                         **
#**                       | '_ \| | | | | __| '_ \| || |_| '_ \| '_ \                        **
#**                       | |_) | |_| | | |_| | | |__   _| | | | | | |                       **
#**                       |_.__/ \__,_|_|\__|_| |_|  |_| |_| |_|_| |_|                       **
#\*==========================================================================================*/

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans

# Initialization
img = mpimg.imread("girl.png");
X = img.reshape((img.shape[0] * img.shape[1], img.shape[2]));
Y = X;

# Test image
def debugInput():
    plt.imshow(img);
    imgplot = plt.imshow(img);
    plt.axis('on');
    plt.show();

# KMC and replace pixels with corresponding centers.
def KMeansClustering(nClusters):
    print("KMC with %d clusters" % nClusters);
    kmeans = KMeans(n_clusters = nClusters).fit(X);
    labels = kmeans.predict(X);
    clcent = kmeans.cluster_centers_;
    Y = np.zeros_like(X)
    for i in range(img.shape[0] * img.shape[1]): Y[i] = clcent[labels[i]];

    print("Done fitting.");
    img5 = Y.reshape((img.shape[0], img.shape[1], img.shape[2]));
    plt.imshow(img5);
    plt.axis('on');
    plt.show();
    print("Done.");

# Mains
debugInput();
for u in [4]: KMeansClustering(u);

