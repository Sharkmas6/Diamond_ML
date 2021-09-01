import sys
import os
import joblib
from sklearn.neighbors import KNeighborsClassifier
from read_data import data, pd, np, plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, plot_confusion_matrix
from scipy.stats import randint, expon, uniform
from xgboost import XGBClassifier
plt.style.use("ggplot")


class n_layer_dist:
    def __init__(self, low, high, n_layers_range, dist=randint):
        self.dist = dist
        self.low, self.high = low, high
        self.n_layers_dist = dist
        self.n_layers_range = n_layers_range

    def rvs(self, *args, **kwargs):
        size = self.n_layers_dist.rvs(*self.n_layers_range, *args, **kwargs)
        layers = self.dist.rvs(self.low, self.high, size=size, *args, **kwargs)
        return tuple(layers)



# system arguments
i = int(sys.argv[1])

# get database
if i <= 5:
    db_kind = "dials"
else:
    db_kind = "3dii"
    i = i - 5
datum = data[db_kind]

workdir = r"/path/to/workdir"
if not os.path.isdir(workdir):
    os.mkdir(workdir)
print(f"DATBASE No. {i} -- DATABASE: {db_kind}")


# use simplified model names
models_names = ["RandomForest", "XGBoost", "KNeighbors", "SVC", "MLP"]
model_name = models_names[i-1]

# get working directories
db_workdir = os.path.join(workdir, db_kind, "confidence_threshold")
db_logfile = os.path.join(db_workdir, f'confidence_{model_name}.log')
if not os.path.isdir(os.path.join(workdir, db_kind)):
    os.mkdir(os.path.join(workdir, db_kind))
if not os.path.isdir(os.path.join(workdir, db_kind, "confidence_threshold")):
    os.mkdir(os.path.join(workdir, db_kind, "confidence_threshold"))


# prepare data
r_etc = ["RMERGE_I", "RMERGE_DIFF_I", "RMEAS_I", "RMEAS_DIFF_I", "RPIM_I", "RPIM_DIFF_I"]
x, y = datum.unpack(drop_col=["DATASET_id", "RESOLUTION_LOW", "RESOLUTION_HIGH", "SPACEGROUP", "SHELXC_CFOM"] + r_etc)

# construct pipelines
seed = 1
print(f"Using seed: {seed}")
scaler = StandardScaler
forest = Pipeline([("scaler", scaler()), ("clf", RandomForestClassifier(class_weight="balanced", random_state=seed))])
xgb = Pipeline([("scaler", scaler()), ("clf", XGBClassifier(class_weight="balanced", random_state=seed))])
kneighbors = Pipeline([("scaler", scaler()), ("clf", KNeighborsClassifier())])
svc = Pipeline([("scaler", scaler()), ("clf", SVC(class_weight="balanced", probability=True, random_state=seed))])
mlp = Pipeline([("scaler", scaler()), ("clf", MLPClassifier(random_state=seed, max_iter=1000))])
models = [forest, xgb, kneighbors, svc, mlp]

# create parameter searches
forest_params = {"clf__criterion": ["gini", "entropy"],
                 "clf__n_estimators": randint(100, 10000),  # number of trees in forest
                 "clf__max_features": randint(2, len(x.columns)),  # max number of features when splitting
                 "clf__min_samples_split": randint(2, 20 + 1),  # min samples per node to induce split
                 "clf__max_depth": randint(5, 20 + 1),  # max number of splits to do
                 "clf__min_samples_leaf": randint(1, 10 + 1),  # min number of samples in a leaf; may set to 1 anyway
                 "clf__max_leaf_nodes": randint(10, 20 + 1)}  # max number of leaves}
xgb_params = {"clf__n_estimators": randint(100, 10000),
              "clf__max_depth": randint(5, 20 + 1),
              "clf__min_child_weight": randint(5, 10 + 1),
              "clf__colsample_bytree": uniform(2/len(x.columns), 1),
              "clf__subsample": uniform(0.1, 1),
              "clf__learning_rate": uniform(0.005, 0.3)}
kneighbors_params = {"clf__weights": ["uniform", "distance"],
                     "clf__n_neighbors": randint(5, 50)}
svc_params = {'clf__C': expon(scale=100),
              'clf__gamma': expon(scale=.1),
              'clf__kernel': ['rbf', "poly"]}
mlp_params = {"clf__alpha": 10.0 ** -np.arange(1, 7),
              "clf__hidden_layer_sizes": n_layer_dist(100, 1000, [1, 5])}
models_params = [forest_params, xgb_params, kneighbors_params, svc_params, mlp_params]


# use randomised search for best possible performance
n_iter = 1000
forest_search = RandomizedSearchCV(forest, forest_params, n_iter=n_iter, scoring="f1", cv=5, random_state=seed)
xgb_search = RandomizedSearchCV(xgb, xgb_params, n_iter=n_iter, scoring="f1", cv=5, random_state=seed)
kneighbors_search = RandomizedSearchCV(kneighbors, kneighbors_params, n_iter=n_iter, scoring="f1", cv=5, random_state=seed)
svc_search = RandomizedSearchCV(svc, svc_params, n_iter=n_iter, scoring="f1", cv=5, random_state=seed)
mlp_search = RandomizedSearchCV(mlp, mlp_params, n_iter=n_iter, scoring="f1", cv=5, random_state=seed)
models_search = [forest_search, xgb_search, kneighbors_search, svc_search, mlp_search]

# choose wanted model based on sys.argv
model = models[i-1]
model_params = models_params[i-1]
model_search = models_search[i-1]


# take only relevant data
mask = x["DATASET_NAME"] == 1
x, y = x[mask], y[mask]
x = x.drop("DATASET_NAME", axis=1)

# drop NaN
mask_nan = x.isna().any(axis=1)
x, y = x[~mask_nan], y[~mask_nan]

# drop outliers
if db_kind == "dials":
    mask = LocalOutlierFactor(contamination=0.4).fit_predict(x)
    mask = mask == 1
    x, y = x.loc[mask, :], y[mask]

# split train/test datasets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.2, stratify=y, random_state=seed)

# run randomized search
model_search.fit(X_train, y_train)
joblib.dump(model_search, os.path.join(db_workdir, f'random_search_{model_name}.pkl'))
joblib.dump(model_search.best_estimator_, os.path.join(db_workdir, f'best_estimator_{model_name}.pkl'))

# get predicted values for classification report and confusion matrix
y_pred = model_search.predict(X_test)
report = classification_report(y_test, y_pred)
matrix = plot_confusion_matrix(model_search, X_test, y_test, normalize="all").confusion_matrix
plt.savefig(os.path.join(db_workdir, f'confusion_matrix_{model_name}'))
confus = pd.DataFrame(matrix, index=["Actual Negative", "Actual Positive"], columns=["Predicted Negative", "Predicted Positive"])


# prepare results
best_params = pd.Series(model_search.best_params_)

# store results
best_params.to_csv(os.path.join(db_workdir, f'best_params_{model_name}.csv'))

log = (f"> Dataset no. 1\n"
       f"> Database: {db_kind}\n"
       f"> Using model: {model_name}\n"
       f"> Best parameters:\n{best_params}\n"
       f"> Best training F1 score: {model_search.best_score_:.2%}"
       f"> Classification report:\n{report}\n"
       f"> Confusion matrix:\n{confus}")

with open(db_logfile, "w") as fhandle:
    fhandle.write(log)
