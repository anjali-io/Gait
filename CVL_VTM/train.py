# train.py
import numpy as np

from model.initialization import initialization
from model.utils import evaluation
from config_cvl import conf
from tsne_vis import tsne_plot


# 🔥 IMPORTANT: do NOT evaluate early
EVAL_START_ITER = 50000   # evaluate only after 50k iters
ENABLE_TSNE = True       # set False for faster runs


def train():
    """
    Train GaitSet + CVL model.
    """
    print("=" * 70)
    print("Starting Training: GaitSet + Cyclic View Learning (CVL)")
    print("=" * 70)

    # Initialize environment, data, and model
    model, save_name = initialization(conf)
    print("Model initialized with name:", save_name)

    # Start training (ITERATION-BASED)
    model.fit()

    print("=" * 70)
    print("Training completed")
    print("=" * 70)

    return model


def test_and_visualize(model):
    """
    Evaluate model on CASIA-B and visualize features using t-SNE.
    Evaluation is SKIPPED if training iterations are too low.
    """
    if model.restore_iter < EVAL_START_ITER:
        print("=" * 70)
        print(
            f"Skipping evaluation ❌ (only {model.restore_iter} iters completed, "
            f"need ≥ {EVAL_START_ITER})"
        )
        print("=" * 70)
        return

    print("=" * 70)
    print("Starting Evaluation on CASIA-B")
    print("=" * 70)

    # Extract features
    features, views, seq_types, labels = model.transform(flag='test')

    # ---- Evaluation ----
    acc = evaluation(
        (features, views, seq_types, labels),
        conf['data']
    )

    print("\n==== Rank-1 Accuracy (Mean over views) ====")
    print("NM (Normal):", acc[0].mean())
    print("BG (Bag):   ", acc[1].mean())
    print("CL (Coat):  ", acc[2].mean())

    # ---- Optional t-SNE Visualization ----
    if not ENABLE_TSNE:
        print("\nSkipping t-SNE visualization (disabled).")
        return

    print("\nGenerating t-SNE visualization (NM sequences only)...")

    mask = np.isin(seq_types, ['nm-01', 'nm-02'])
    features_nm = features[mask]
    labels_nm = np.array(labels)[mask]
    views_nm = np.array(views)[mask]

    tsne_plot(
        features_nm,
        labels_nm,
        views_nm,
        title="t-SNE of Gait Features (GaitSet + CVL)"
    )


if __name__ == "__main__":
    # -------- TRAIN --------
    model = train()

    # -------- TEST + VISUALIZE (SAFE) --------
    test_and_visualize(model)

