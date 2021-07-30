from matplotlib.patches import Patch
from sklearn.cluster import KMeans
from read_data import Data, pd, plt
from collections import OrderedDict


class GroupedData(Data):
    '''
    Class to hold grouped feature/label data
    '''
    def __iter__(self):
        return iter(self.groups.items())

    def sort(self, sort_order, inplace=False):
        sorted_dict = OrderedDict({key: self[key] for key in sort_order})  # sort groups by given order

        if inplace:
            self.groups = sorted_dict  # change own groups
        else:
            new = GroupedData(self.x, self.y, name=self.name)  # initial copy of self
            new.groups = sorted_dict
            return new

    def __getitem__(self, item):
        assert item in self.groups, "Desired group not found"
        return self.groups[item]

    def __setitem__(self, key, value):
        assert key in self.groups, "Desired group not found"
        self.groups[key] = value

    def __len__(self):
        return len(self.groups)

    def __str__(self):
        try:
            return f"Grouped Data {self.x.shape} with {len(self)} groups."
        except AttributeError:
            raise Exception("No groups set yet, call .group() method first")

    def group(self, n_groups, names=None):
        '''
        :param n_groups: int array/series stating which group each datapoint belongs to
        :param names: container specifying the names of each group
        :return: grouped self
        '''

        grps = pd.unique(n_groups)  # pd.unique preserves order, unlike np.unique

        # custom group names
        proper_names = False
        if names is not None:
            assert len(grps) == len(names), "Must give same amount of names as unique data groups"
            proper_names = True
        else:
            names = grps  # change to dummy variable if no names given

        self.groups = {}
        for grp, name in zip(grps, names):
            # find group data
            mask = n_groups == grp
            x_temp = self.x.loc[mask, :] if self.x.ndim == 2 else self.x.loc[mask]
            y_temp = self.y[mask]

            # create groups and store in own dict
            key = name if proper_names else grp
            self.groups[key] = Data(x_temp, y_temp, name=key)

        return self


class GroupedSpacegroup(GroupedData):
    '''
    Class to hold feature/label data grouped by descending Spacegroup mean success rate
    '''
    def group(self, n, names=None, auto_sort=True):
        '''
        :param n: number of groups to cluster data by
        :param names: list of names to give to groups
        :param auto_sort: whether to automatically sort groups to order given in names parameter
        :return: KMeans model used to group, predicted clusters, grouped self
        '''

        # get spacegroup clusters
        model, pred, agg_sorted = cluster_spacegroups(self.union, n)
        clusters = self.x["SPACEGROUP"].map(pred)

        # sort group names by data order - needed for proper group name-data pairing
        names_map = pd.Series(names, pred.drop_duplicates())  # group number (KMeans) <-> given names
        data_sorted_names = clusters.drop_duplicates().map(names_map)

        # group data based on clusters
        super(GroupedSpacegroup, self).group(clusters, names=data_sorted_names)

        # automatically sort
        if names is not None and auto_sort:
            super(GroupedSpacegroup, self).sort(names, inplace=True)

        return model, clusters, self



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
