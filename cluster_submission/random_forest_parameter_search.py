import sys
import os
import joblib
from read_data import data, pd, plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, plot_confusion_matrix
from scipy.stats import randint
plt.style.use("ggplot")



# system arguments
i = int(sys.argv[1])

# get database
if i <= 100:
    db_kind = "dials"
else:
    db_kind = "3dii"
    i = i - 100
datum = data[db_kind]

workdir = f"/path/to/workdir/dataset_{i}"
if not os.path.isdir(workdir):
    os.mkdir(workdir)
print(f"DATASET No. {i} -- DATABASE: {db_kind}")


# get working directories
db_workdir = os.path.join(workdir, db_kind)
db_logfile = os.path.join(db_workdir, 'classifier.log')
if not os.path.isdir(db_workdir):
    os.mkdir(db_workdir)
r_etc = ["RMERGE_I", "RMERGE_DIFF_I", "RMEAS_I", "RMEAS_DIFF_I", "RPIM_I", "RPIM_DIFF_I"]
cols2drop = ["DATASET_id", "RESOLUTION_LOW", "RESOLUTION_HIGH", "SPACEGROUP", "SHELXC_CFOM"]
x, y = datum.unpack(drop_col=cols2drop + r_etc)


# take only relevant data
mask = x["DATASET_NAME"] == i
x, y = x[mask], y[mask]
x = x.drop("DATASET_NAME", axis=1)

# drop NaN
mask_nan = x.isna().any(axis=1)
x, y = x[~mask_nan], y[~mask_nan]

# split train/test datasets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.2, stratify=y)

# set up randomized search
param_rand = {"clf__class_weight": [None, "balanced"],
              "clf__criterion": ["gini", "entropy"],  # metric to judge reduction of impurity
              "clf__n_estimators": randint(100, 10000),  # number of trees in forest
              "clf__max_features": randint(2, len(X_train.columns) + 1),  # max number of features when splitting
              "clf__min_samples_split": randint(2, 20 + 1),  # min samples per node to induce split
              "clf__max_depth": randint(5, 10 + 1),  # max number of splits to do
              "clf__min_samples_leaf": randint(1, 20 + 1),  # min number of samples in a leaf; may set to 1 anyway
              "clf__max_leaf_nodes": randint(10, 20 + 1)}  # max number of leaves

# run randomized search
n_iter = 1000
clf = RandomForestClassifier()
pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
rand_search = RandomizedSearchCV(pipe, param_rand, n_iter=n_iter, cv=5, scoring="f1", n_jobs=1)
rand_search.fit(X_train, y_train)
joblib.dump(rand_search, os.path.join(db_workdir, 'random_search.pkl'))
joblib.dump(rand_search.best_estimator_, os.path.join(db_workdir, 'best_estimator.pkl'))

# get and save confusion matrix
matrix = plot_confusion_matrix(rand_search, X_test, y_test, normalize="all").confusion_matrix
plt.savefig(os.path.join(db_workdir, 'confusion_matrix'))

# get predicted values for classification report and confusion matrix
y_pred = rand_search.predict(X_test)
report = classification_report(y_test, y_pred)
confus = pd.DataFrame(matrix, index=["Actual Negative", "Actual Positive"], columns=["Predicted Negative", "Predicted Positive"])


# prepare results
best_params = pd.Series(rand_search.best_params_)
topfeat = pd.Series(rand_search.best_estimator_['clf'].feature_importances_, index=X_train.columns)
topfeat_sorted = topfeat.sort_values(ascending=False, key=lambda k: abs(k))

# store results
best_params.to_csv(os.path.join(db_workdir, 'best_params.csv'))
topfeat.to_csv(os.path.join(db_workdir, 'topfeat.csv'))
topfeat_sorted.to_csv(os.path.join(db_workdir, 'topfeat_sorted.csv'))

# create log file
log = (f"> Dataset no. {i}\n"
       f"> Database: {db_kind}\n"
       f"> Using model: {clf}\n"
       f"> Best parameters:\n{best_params}\n"
       f"> Best training F1 score: {rand_search.best_score_:.2%}\n"
       f"> Classification report:\n{report}\n"
       f"> Confusion matrix:\n{confus}\n"
       f"> Feature importances:\n{topfeat_sorted}")

with open(db_logfile, "w") as fhandle:
    fhandle.write(log)