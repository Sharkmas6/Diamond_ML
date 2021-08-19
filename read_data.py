import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from f_double_prime import FDoublePrime
pd.set_option("display.width", None)


def sql_to_df(db_file, *tb_names, search_query=None):
    '''
    function to connect to database and convert datum to pandas DataFrame,
    use #TB_NAME# to refer to table name
    '''
    # make connection
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()

    for tb_name in tb_names:
        # query table
        cur.execute(search_query.replace("#TB_NAME#", tb_name))
        result = cur.fetchall()

        # fetch column names
        cur.execute(rf"PRAGMA table_info({tb_name})")
        cols = [info[1] for info in cur.fetchall()]

        # create DataFrame
        df = pd.DataFrame(result, columns=cols)
        df.set_index("id", inplace=True)
        df.name = tb_name

        yield df


class Data:
    '''
    Class to hold feature/label data
    '''
    def __init__(self, data, targets, name=None):
        try:
            assert data.shape[0] == len(targets), "Data and targets should be of same size"
        except AttributeError:  # case not a numpy or pandas structure, e.g. list or tuple
            raise TypeError("This class only works with numpy/pandas data structures, "
                            "please restructure your data into one of these and try again")

        self.x = data
        self.y = targets
        self.union = pd.concat([self.x, self.y], axis=1)
        self.name = name
        self._dict = {"x": self.x, "y": self.y, "union": self.union}

    def __getitem__(self, item):
        return self._dict[item]

    def __setitem__(self, key, value):
        self._dict[key] = value

    def __str__(self):
        return f"{self.name} data with shape {self.x.shape} - {np.unique(self.y).size} categories"

    def unpack(self, union=False, drop_col=None):
        '''
        Easy function to unpack data/features and targets/labels in a single statement.
        Union of these also available.
        :param union: Whether to include returning union or not
        :param drop_col: Try dropping mentioned columns
        :return: data, targets (and union if union=True) with "drop" columns dropped, if any
        '''
        x_final, y_final, union_final = self.x, self.y, self.union

        # drop specified columns, if possible
        if drop_col:
            try:
                x_final = x_final.drop(drop_col, axis=1)
                union_final = union_final.drop(drop_col, axis=1)
            except KeyError:
                print("Unable to find columns intended to drop, resuming..")

        # return x, y (and union if specified)
        if union:
            return x_final, y_final, union_final
        else:
            return x_final, y_final


tables = [r"SWEEP_STATS", r"EP_STATS", r"PDB_DATA", r"DATASET_INFO"]
query = r"SELECT * FROM #TB_NAME#"
names = ["dials", "3dii"]
db_paths = [fr"D:\Diamond\cesar_project_{name}.db" for name in names]  # locations on my machine
data = {}

for name, db_path in zip(names, db_paths):

    sweep_full, ep_full, pdb_full, dataset_full = [i for i in sql_to_df(db_path, *tables, search_query=query)]  # old x, y

    # cut to subset
    sample_frac = 1
    sweep, ep = sweep_full, ep_full#.sample(frac=sample_frac), ep_full.sample(frac=sample_frac)


    x = sweep
    # define label
    y = ep["IS_SUCCESS"]
    y.index = ep["DATASET_id"]
    y = x["DATASET_id"].map(y)
    y.name = "IS_SUCCESS"

    # add ep stats to x
    mapped = []
    for col_name in ["SOLVENT_CONTENT", "NUMBER_SITES"]:
        ep_temp = ep.loc[:, col_name]
        ep_temp.index = ep["DATASET_id"]
        ep_temp = x["DATASET_id"].map(ep_temp)
        ep_temp.name = col_name

        mapped.append(ep_temp)
    x = pd.concat([x] + mapped, axis=1)

    # replace wavelength by f''
    x["WAVELENGTH"] = FDoublePrime(x["WAVELENGTH"].values)
    x.rename(columns={"WAVELENGTH": "F''"}, inplace=True)

    # add TOT_MOLWEIGHT = NMOL * MOLWEIGHT to database
    molweight_map = pd.Series(pdb_full["MOLWEIGHT"].values, index=pdb_full["PDB_CODE"])
    molweight = dataset_full["PDB_CODE"].map(molweight_map)
    totweight = molweight * dataset_full["NMOL"]
    x["TOT_MOLWEIGHT"] = x["DATASET_id"].map(totweight)

    # add DATASET_NAME
    cols = ["DATASET_NAME", "RESOLUTION_LOW", "RESOLUTION_HIGH"]
    for col in cols:
        x[col] = x["DATASET_id"].map(dataset_full[col])
    # fix unrestricted (None) datasets
    x["DATASET_NAME"] = x["DATASET_NAME"].apply(lambda name: int(name[8:]))
    '''low_none = (x["DATASET_NAME"] - 1) // 10 == 0  # first 11 values
    high_none = x["DATASET_NAME"] % 10 == 1
    for col, mask in zip(["RESOLUTION_LOW", "RESOLUTION_HIGH"], [low_none, high_none]):
        x.loc[mask, col] = None'''

    # limit to common datasets
    valid_ids = set.intersection(set(x["DATASET_id"]), set(ep["DATASET_id"]))
    a = x["DATASET_id"].isin(valid_ids)
    x = x[x["DATASET_id"].isin(valid_ids)]
    y = y[y.index.map(sweep["DATASET_id"]).isin(valid_ids)]

    # filter out undetermined values
    mask = y.isin([1, 0])
    y = y[mask]
    x = x[mask]
    x.drop(columns="WAVE_NAME", inplace=True)

    # optimise dtypes to save memory
    x["SPACEGROUP"] = x["SPACEGROUP"].astype("category")
    y = y.astype("bool")

    # add results to dict
    data[name] = Data(x, y, name=name)


def truncate_data(x=x, y=y, n=None):
    # limit to equal negative/positive labels
    n = np.count_nonzero(y) if not n else n
    lose = y == 0
    lose_sample = y[lose].sample(n=n)
    lose_sample_mask = lose_sample == 0
    win = y == 1
    win_sample = y[win].sample(n=n)
    win_sample_mask = win_sample == 1

    mask = y.astype(bool).copy()
    mask.loc[:] = False
    mask.update(lose_sample_mask)
    mask.update(win_sample_mask)
    x_redu, y_redu = x[mask], y[mask]

    return x_redu, y_redu


def plot_original_data(col1_name, col2_name, data, target, alpha=1, show=True, **fig_kwargs):
    # color each group
    cmap = {0: "tab:red", 1: "tab:green"}
    c = target.map(cmap)
    patches = [Patch(color=c, label=l) for c, l in zip(cmap.values(), ["Failure", "Success"])]

    # plot normal data
    fig, ax = plt.subplots(**fig_kwargs)
    x_plot, y_plot = data.loc[:, col1_name], data.loc[:, col2_name]

    ax.scatter(x_plot, y_plot, c=c, alpha=alpha)

    ax.set_xlabel(x_plot.name)
    ax.set_ylabel(y_plot.name)
    ax.set_title("Original Data")
    ax.legend(handles=patches, title="Groups")

    if show:
        plt.show()

    return (fig, ax), (cmap, c, patches)
