from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

if __name__ == "__main__":
    pass


def cluster_spacegroups(union, n_clusters):
    # group by success and sort
    agg_sorted = union.groupby("SPACEGROUP").agg(["mean", "sem"])\
        .sort_values(("IS_SUCCESS", "mean"), ascending=False)

    # ML grouping
    model = KMeans(n_clusters=n_clusters)
    pred = model.fit_predict(pd.DataFrame(agg_sorted["IS_SUCCESS", "mean"]))
    pred = pd.Series(pred, index=agg_sorted.index)

    return model, pred, agg_sorted


def avg_success_bar_plot(agg_sorted, labels, clusters, show=True, sort_legend=True):
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.xticks(rotation=90)

    # colormap
    cmap = plt.cm.get_cmap("Dark2")
    c = clusters.map(cmap)
    ax.bar(agg_sorted.index, agg_sorted["IS_SUCCESS", "mean"],
           yerr=agg_sorted["IS_SUCCESS", "sem"], color=c)
    ax.hlines(labels.mean(), 0, agg_sorted.index.size - 1, colors="brown", label="Mean")

    # legend
    patches = [Patch(color=color, label=label) for color, label in \
               zip(c.drop_duplicates(), ["High", "Average", "Low", "Null"])]
    ax.legend(handles=patches, title="Groups")

    # text
    ax.set_title(f"Sorted Spacegroup Success Distribution with {len(clusters.drop_duplicates())} groups")
    ax.set_xlabel("Space Group")
    ax.set_ylabel("Average Success Rate")

    if show:
        plt.show()

    return fig, ax, c

