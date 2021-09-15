# Diamond_ML

---

## Summer Project for Diamond Light Source

Python scripts/apps for Diamond Light Source's Summer Placement.
Researching optimal features for correct protein identification from electron density maps, obtained using EP.

---

## Main Analysis

The main analysis of this project is (almost) all stored in Jupyter Notebooks, found in Diamond_ML/jupyter/.
It's in these notebooks where most details of my exploration and analysis are stored.
In each of them I explore a "unique" concept (e.g. correlations, feature importances, PCA/LDA).

In the **first half** of this project (Diamond_ML/jupyter/outdated/), initial **EDA**, **PCA/LDA** analysis and **spacegroup grouping** was attempted.

In the **second half** (Diamond_ML/jupyter/updated/), the data used in these was expanded, and more "proper" methods of analysis were used.
Here the **low/high resolution impacts** on EP success rate, **dimensionality reduction**, **outlier removal**, **feature importance** per resolution limits, and 
**model comparison** with varying **confidence thresholds** were all explored.

---

## Final EP Model

After this analysis, I created an **EPModel** class intended to serve as an all-in-one package to predict the outcome of EP.
This class currently works for either *DIALS* or *XDS* (aka *3DII*), and is able to create a pipeline with 4 essential steps:

1. Remove outliers
2. Select relevant features
3. Scale remaining data
4. Train classifier

The specifics of which are determined at the time of creation.
These steps are then all done automatically when calling the fit, predict, or test methods.

Using the Data class found in `read_data.py`, an example usage would look as such:

```
from read_data import data
from sklearn.model_selection import train_test_split

# fetch data
db_kind = "dials"
datum = data[db_kind]

# prepare data
X, y = datum.unpack(drop_col=["SHELXC_CFOM", "SPACEGROUP"])
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=.2)

# prepare model params
model_params = {hidden_layer_sizes: (800, 400, 600),
                alpha: 0.001}

# create EP Model
ep = EPModel(data_pipeline=db_kind,
             model_kind="MLP",
             model_params=model_params)
             
# fit model
ep.fit(X_train, y_train)

# get test scores dict
scores = mdl.test(X_test, y_test, outlier_contam=0.4)

# show info summary and test scores
print(mdl.summary())
print(pd.Series(scores))
```

Which would output

```


```

In this example, data is taken from the `read_data` script, which is then separated into

After this comes a code block


    from sklearn.neighbors import KNeighborsClassifier
    
    
    a = 2
    print(a*2)


More text

Link to [website](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EllipticEnvelope.html?highlight=elliptic#sklearn.covariance.EllipticEnvelope)

---

