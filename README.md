# Diamond_ML

---

## Summer Project for Diamond Light Source

This repo stores Python scripts and apps for my Summer Placement at Diamond Light Source. 

The goals of this placement were to:

1. Research the impact of low/high resolution limits on EP
2. Create a machine learning model capable of predicting successful protein identification from a set of initial quality metrics.

---

## Requirements

Due to the usage of f-strings, this project requires _at least_ **Python 3.6**.

It also made strong use of `numpy`, `pandas`, `matplotlib`, `scipy` and `scikit-learn`, some combination of which is almost always required for recreation and/or running of the scripts presented. 

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

After this analysis, I created an **EPModel** class intended to serve as an all-in-one package to predict the outcome of EP, found on `FinalModel/compound_model.py`.
This class currently works for either *DIALS* or *XDS* (aka *3DII*), and is able to create a pipeline with 4 essential steps:

1. Remove **outliers**
2. Select relevant **features**
3. **Scale** remaining data
4. **Train** classifier

The specifics of which are determined at the time of creation.
These steps are then all done automatically when calling the fit, predict, or test methods.

Using the Data class found in `read_data.py`, an example usage would look as such:

```
import pandas as pd
from read_data import data
from compound_model import EPModel
from sklearn.model_selection import train_test_split


# fetch data
db_kind = "dials"
datum = data[db_kind]
X, y = datum.unpack(drop_col=["SHELXC_CFOM", "SPACEGROUP", "PDB_CODE"])

# take only untouched data
mask = X["DATASET_NAME"] == 1
X, y, = X[mask], y[mask]
X = X.drop(["DATASET_NAME"], axis=1)

# separate into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=.2)

# prepare model params
classifier_params = {"hidden_layer_sizes": (800, 400, 600),
                     "alpha": 0.001}

# create EP Model
ep = EPModel(data_pipeline=db_kind,
             classifier_kind="MLP",
             classifier_params=classifier_params)

# fit model
ep.fit(X_train, y_train)

# get test scores dict
scores = ep.test(X_test, y_test)

# show info summary and test scores
print()
print(ep.summary(), end="\n" * 2)
print(pd.Series(scores))
```

Wich would output:

```
Fitting model on 994 samples, 35 features...
Removed 398 samples (40.04%) as outliers
Testing model on 207 samples
Predicting outcome of 207 samples...

Experimental Phasing prediction model for DIALS data, with settings:
> Predictor: MLPClassifier(alpha=0.001, hidden_layer_sizes=(800, 400, 600))
> Scaler: StandardScaler()
> Feature selector: ColumnTransformer(transformers=[('feat_select', 'passthrough',
                                 ['ANOMALOUS_CORRELATION', 'DIFF_I',
                                  'SOLVENT_CONTENT', 'ANOMALOUS_SLOPE',
                                  'SHELXC_SIGNAL', 'NUMBER_SITES',
                                  'HIGH_RES_LIMIT'])])
> Outlier removal: LocalOutlierFactor()
> Default contamination: 0.4
> Model trained: True

accuracy     0.705314
f1           0.596026
mcc          0.365844
precision    0.625000
recall       0.569620
dtype: float64
```

In this example, data is taken from the `read_data` script, which is then separated isolated and separated into training and testing datasets.
After this, it's a matter of creating an EPModel object with the specified parameters, fitting and then testing.
A summary of the model info was also shown as an example.

For more information on further customisation, see below.

---

## CMD Interface

For ease of usage, a command-line interface for this project was also created.
The script governing this interface is `FinalModel/parser.py`.

To run, this interface requires `numpy`, `pandas`, `scikit-learn`, `joblib`, and `compound_model`.
It also makes use of `argparse` and `ast`, which are usually builtins. 

To access it, _cd_ to the location of `parser.py` and call it through Python (assuming it's already in PATH).

```
python parser.py [-commands] [arguments]
```

The `-h` flag indicates all the available commands, their descriptions, usage, and default values (if relevant).

The main commands are:

- `-fit`: Requires `X_TRAIN` for features and `Y_TRAIN` for targets/labels.
  - Trains model and dumps (stores) it on desired location, specified by --model_loc or --output. Defaults to `EPModel.pkl`.
- `-predict`: Requires `X_PRED` for features.
  - Use already trained model to predict EP outcome.
  - Stores result on --output. Defaults to `predictions.csv`.
- `-test`: Requires `X_TEST` for features and `Y_TEST` for targets/labels.
  - Test already trained model and shows resulting scores/metrics.
  - Outputs results to --output as csv file. Defaults to `test_results.csv`.
- `-summary`: Load EPModel and list summary.
  - Stored in --output. None by default (no storage).
  - Can be used together with fit, to show summary of just trained model.
  
**ATTENTION:**
> The data given to `-fit`, `-predict`, and `-test` must specify the location of _csv_ files, and include a "dummy" header and index lines,
from which feature names will be inferred if the default feature set is chosen.

### Basic usage

A very simple usage of this interface would be training a default model on some datasets.
If our example file layout looked as such:

```
> Dir
---> parser.py
---> compound_model.py
---> x_train.csv
---> y_train.csv
---> x_test.csv
---> y_test.csv
```

Then a basic fitting command would be:

```
python parser.py -fit x_train.csv y_train.csv -summary
```

Here summary is optional, but was included for demonstration purposes. This would output

```
No custom model parameters found, using default parameters
Consider specifying the model parameters to be used, either through --classifier_params or through adding a classifier_params.txt file in the same directory as this file

Fitting model on 994 samples, 35 features...
Removed 398 samples (40.04%) as outliers
Stored results in EPModel.pkl

Experimental Phasing prediction model for DIALS data, with settings:
> Predictor: MLPClassifier()
> Scaler: StandardScaler()
> Feature selector: ColumnTransformer(transformers=[('feat_select', 'passthrough',
                                 ['ANOMALOUS_CORRELATION', 'DIFF_I',
                                  'SOLVENT_CONTENT', 'ANOMALOUS_SLOPE',
                                  'SHELXC_SIGNAL', 'NUMBER_SITES',
                                  'HIGH_RES_LIMIT'])])
> Outlier removal: LocalOutlierFactor()
> Default contamination: 0.4
> Model trained: True
```

This would create an `EPModel.pkl` file in the same directory, which could easily be tested with:

```
python parser.py -test x_test.csv y_test.csv
```

Outputting:

```
Testing model on 207 samples
Predicting outcome of 207 samples...
accuracy     0.797101
f1           0.704225
mcc          0.560979
precision    0.793651
recall       0.632911
dtype: float64
Stored results in test_results.csv
```

Whose results would be stored in `test_results.csv`, as mentioned.

For more advanced usage and customisation, check below.

---

## Advanced Usage

### EPModel Parameters

EPModel doesn't currently possess many parameters, but it still allows for reasonable customisation.
When created, these are the parameters available:

- `data_pipeline`: Either `dials`, `xds`, or `3dii`. Specifies which pipeline the data was processed with. 
This changes the default contamination and features used. xds is an alias for 3dii.
- `classifier_kind`: Either `MLP`, `RandomForest`, or `SVC`. Specifies which classifier to use.
- `classifier_params`: Dictionary specifying the specific classifier parameters to customise `classifier_kind` with.
- `feat_select_type`: If `kbest`, use SelectKBest feature selector, with parameters set by `feat_select_params`.
Else use default, pre-selected set of features, depending on `data_pipeline`.
- `feat_select_params`: Dictionary specifying SelectKBest parameters. Main usage of specifying k.
Defaults to {"k": 7}.
- `metrics`: Metrics to calculate during test.
Can be any combination of `accuracy`, `f1`, `mcc`, `precision`, and `recall`. Uses all by default.
- `outlier_kind`: Either `LOF`, `IsolationForest`, or `EllipticEnvelope`. Specifies method to use when removing outliers.
Defaults to _LOF_.
- `default_contam`: Outlier contamination to use when contamination isn't specified during fit/predict/test calls.
Defaults to 0.4 for _DIALS_, and 0 for _XDS_.
- `verbose`: Either 1 or 0. If 1, print short status messages during fitting/predicting/testing. If 0, don't print these.
Defaults to 1.

---

### EPModel Methods

EPModel currently has 5 methods. These methods, and their parameters, are:

- `filter_outliers`: Filter dataset(s) according to given contamination.
If contamination not in [0, 1], return untouched dataset(s).
  - `X`: Main dataset in which to detect outliers and obtain a mask from. Will always be filtered and returned if possible
  - `y`: Secondary dataset to filter. Will not be used for outlier detection, but will be filtered and returned whenever possible.
  - `contamination`: How much of the data to consider to be outliers. Simply, fraction of data to be removed.
- `fit`: Fit current model to training dataset.
  - `X`: Training features. 2D required.
  - `y`: Training targets/labels. 1D required.
  - `outlier_contam`: Contamination used when removing outliers. None for default values.
  - `fit_params`: Any additional parameters to be passed to the underlying pipeline's fit method.
- `predict`: Predict EP outcome using given features. Model must have been previously trained.
  - `X`: Feature dataset to use for prediction.
  - `confidence`: Confidence threshold. Minimum certainty for model prediction decision
                           (e.g. a confidence of 80% means that the samples predicted to succeed will be
                           those 80% or more likely to succeed).
  - `outlier_contam`: Contamination used when removing outliers. None for default values.
  - `*args` & `**kwargs`: (Keyword) Arguments to be passed to the underlying pipeline's predict_proba method.
- `test`: Test model on test dataset. Model must have been previously trained.
  - `X_test`: Feature dataset to use for prediction.
  - `y_test`: Target/label dataset to use for comparison with predictions and obtaining scores.
  - `confidence`: See predict's confidence.
  - `outlier_contam`: See fit's outlier_contam
  - `scoring`: Scoring metrics to calculate and return. Any combination from 'accuracy', 'f1', 'mcc', 'precision', 'recall'.
             Uses all by default.
- `summary`: Obtain summary of classifier, scaler, feature selector, etc.

---

## CMD Interface Customisation

This CMD Interface serves as a "wrapper", of sorts, of the EPModel class, allowing it to be used outside a script, through the command-line.
As such, it allows for the customisation of most parameters just now described.

### Specifying Classifier Parameters

If no classifier parameters are specified, then the default scikit-learn parameters will be used.
This often leads to too basic, under/overfit models, which is undesirable.

To change this, during fitting, the model looks for a file called `classifier_params.txt`, in which the model kind and parameters are specified.
This is **not** a _csv_ file, and instead uses a colon (:) to indicate assignment. For example:

```
classifier_kind:MLP
hidden_layer_sizes:(800, 400, 600)
alpha:0.001
learning_rate:adaptive
```

Indicates EPModel to select `classifier_kind=MLP`, with `classifier_params={hidden_layer_sizes:(800, 400, 600), alpha:0.001, learning_rate:"adaptive"}`.
This can also be used with RandomForest or SVC, as long as the given parameters are available for the underlying model (check [scikit-learn's website](https://scikit-learn.org/stable/index.html))

A custom `classifier_params.txt` location and name can also be specified through the --classifier_params argument.
This tells the model where to look for a `classifier_params.txt`-like file.

### Specifying custom locations

By default, `-fit` stores a model in `EPModel.pkl`, the same place from where `-predict`, `-test`, and `-summary` try to fetch a model.
However this model location can be altered with `--model_loc`.

Also by default, `-fit`, `-predict`, and `-test` store their results in `EPModel.pkl`, `predictions.csv`, and `test_results.csv`, respectively, in the same directory as `parser.py`.
But this can also be altered through the `--output` command, indicating where to store the output.

`--output` can also be set to _none_ (capitalisation irrelevant) for no storage. This is useful for debugging and/or checking if the program is running nicely without creating any files.

If both `--model_loc` and `--output` are specified, and `-fit` is called, the latter will be used. 

### Additional optional arguments

A few other arguments can be set. These are:

- `--pipeline`: Either `dials`, `xds`, or `3dii` (latter two are equal). Set data origin.
- `--kbest`: If _int_, use SelectKBest feature selection method with k best features.
If 0, use pre-selected set of features, depending on `--data_pipeline`.
- `--outlier_model`: Either `LocalOutlierFactor`, `IsolationForest`, or `EllipticEnvelope` (shortened to `LOF`, `IF`, `EE`, respectively).
Choose outlier removal method/model to use. Capitalisation irrelevant.
- `--default_contam`: Default contamination of EPModel. Defaults to 0.4 for DIALS, and 0 for XDS.

---

## Acknowledgements

I'd like to thank my supervisors Melanie Vollmar and Filomeno Sanchez, not only for giving me a chance, but also for all their support and guidance throughout this project.

