import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def tsne_plot(features, labels, views=None, title="t-SNE Visualization", max_samples=1000):
    """
    t-SNE visualization for gait embeddings.

    Args:
        features: (N, D) numpy array
        labels: (N,) identity labels
        views: (optional) view labels (ignored, for compatibility)
        title: plot title
        max_samples: subsample for speed
    """

    features = np.asarray(features)
    labels = np.asarray(labels)

    if len(features) > max_samples:
        idx = np.random.choice(len(features), max_samples, replace=False)
        features = features[idx]
        labels = labels[idx]

    print("Running t-SNE...")

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate='auto',
        max_iter=1000,
        init='pca',
        random_state=42
    )

    emb = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        emb[:, 0],
        emb[:, 1],
        c=labels,
        cmap='tab20',
        s=6,
        alpha=0.8
    )

    plt.colorbar(scatter)
    plt.title(title)
    plt.tight_layout()
    plt.show()

