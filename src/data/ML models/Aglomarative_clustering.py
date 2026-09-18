from src.data.load_data import load_data
from src.data.preprocess import *
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

def create_model(k):
    model = AgglomerativeClustering(
        n_clusters=k,
        linkage="ward",
    )
    return model

def train_model(model, X):
    model.fit(X)
    print("\nAgglomerative Clustering completed!")
    return model

def evaluate_model(model, X):
    labels = model.labels_
    score = silhouette_score(X, labels)
    print("\nSilhouette Coefficient: ")
    print(score)
    return labels

def display_dendrogram(X, cut_distance):
    dendrogram(
        linkage,
        truncate_mode="lastp",
        p=30
    )
    plt.axhline(y=cut_distance, color="r", linestyle="--")
    plt.title("Agglomerative Clustering Dendrogram")


