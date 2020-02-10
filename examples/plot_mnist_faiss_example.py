"""
UMAP on the MNIST Digits dataset
--------------------------------

A simple example demonstrating how to use UMAP on a larger
dataset such as MNIST. We first pull the MNIST dataset and
then use UMAP to reduce it to only 2-dimensions for
easy visualisation.

Note that UMAP manages to both group the individual digit
classes, but also to retain the overall global structure
among the different digit classes -- keeping 1 far from
0, and grouping triplets of 3,5,8 and 4,7,9 which can
blend into one another in some cases.
"""
import umap
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import seaborn as sns

import os.path
import faiss
import numpy as np

mnist = fetch_openml("mnist_784", version=1)
sns.set(context="paper", style="white")

index = faiss.IndexHNSWSQ(784, faiss.ScalarQuantizer.QT_8bit, 32)
index.verbose = True
faiss_index_file = 'faiss.index'
if os.path.exists(faiss_index_file):
    print('load existing index from %s' % faiss_index_file)
    index = faiss.read_index(faiss_index_file,faiss.IO_FLAG_MMAP)
    index.hnsw.efSearch = 256
else:
    # build lossy faiss index 
    print('build new index and save to %s' % faiss_index_file)
    index.hnsw.efConstruction = 40
    data = np.ascontiguousarray(mnist.data, dtype=np.float32)
    # we no longer need mnist data in its original form
    print('train index...')
    index.train(data)
    print('add vectors to index...')
    index.add(data)
    print('save...')
    faiss.write_index(index, faiss_index_file)

reducer = umap.UMAP(random_state=42, init="random", verbose=True, n_epochs=200)
embedding = reducer.fit_faiss_transform(index)
#embedding = reducer.fit_transform(mnist.data)

fig, ax = plt.subplots(figsize=(12, 10))
color = mnist.target.astype(int)
plt.scatter(embedding[:, 0], embedding[:, 1], c=color, cmap="Spectral", s=0.1)
plt.setp(ax, xticks=[], yticks=[])
plt.title("MNIST data embedded into two dimensions by UMAP", fontsize=18)

plt.show()
