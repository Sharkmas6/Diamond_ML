import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef, precision_score, recall_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope


class EPModel():
    '''
    General class used to predict outcome of EP based on given metrics
    '''

    pipeline_kinds_ = ["dials", "xds", "3dii"]
    available_models_ = {"SVC": SVC,
                         "RandomForest": RandomForestClassifier,
                         "MLP": MLPClassifier}
    feat_selections_ = {"Default": {"dials": ["ANOMALOUS_CORRELATION", "DIFF_I", "SOLVENT_CONTENT", "ANOMALOUS_SLOPE",
                                              "SHELXC_SIGNAL", "NUMBER_SITES", "HIGH_RES_LIMIT"],
                                    "3dii": ["TOT_MOLWEIGHT", "ANOMALOUS_CORRELATION", "MULTIPLICITY", "NUMBER_SITES",
                                             "TOTAL_UNIQUE_OBSERVATIONS", "HIGH_RES_LIMIT"]},
                        "KBest": SelectKBest}
    feat_selections_["Default"]["xds"] = feat_selections_["Default"]["3dii"]
    available_metrics_ = {"accuracy": accuracy_score,
                          "f1": f1_score,
                          "mcc": matthews_corrcoef,
                          "precision": precision_score,
                          "recall": recall_score}
    available_outliers_ = {"LOF": LocalOutlierFactor,
                           "IsolationForest": IsolationForest,
                           "EllipticEnvelope": EllipticEnvelope}

    def __init__(self, data_pipeline="dials", model_kind="MLP", feat_select_type="default", model_params={},
                 feat_select_params={"k": 7}, metrics=["accuracy", "f1", "mcc", "precision", "recall"],
                 outlier_kind="LOF"):
        assert data_pipeline in self.pipeline_kinds_, "Model only available for DIALS and XDS (3DII)"
        assert model_kind in self.available_models_.keys(), f"Please select one of: {list(self.available_models_.keys())}"

        # store classifier and scaler, combine into pipeline
        self.clf = self.available_models_[model_kind](**model_params)
        self.scaler = StandardScaler()
        self.model = Pipeline([("scaler", self.scaler), ("clf", self.clf)])
        self.is_trained_ = False

        # create column transformer depending on feat_selection kind
        if feat_select_type == "kbest":
            # select k best features
            self.feat_selector = SelectKBest(**feat_select_params)
            self.model.steps.insert(0, ("feat_select", self.feat_selector))
        elif feat_select_type == "default":
            # select predefined features
            trsfs = [("feat_select", "passthrough", self.feat_selections_["Default"][data_pipeline])]
            self.feat_selector = ColumnTransformer(trsfs)
            self.model.steps.insert(0, ("feat_select", self.feat_selector))

        # store settings
        self.data_pipeline = data_pipeline
        self.feat_select_type = feat_select_type
        self.metrics = metrics
        self.outlier = self.available_outliers_[outlier_kind]

    def filter_outliers(self, X, y=None, contamination=0.4):
        if contamination <= 0 or contamination >= 1:
            if y is not None:
                return X, y
            else:
                return X

        if self.data_pipeline == "dials":
            # only filter if dials data
            mask = self.outlier(contamination=contamination).fit_predict(X) == 1
            print(f"Removed {mask.size-mask.sum()} samples ({1-mask.sum()/mask.size:.2%}) as outliers")
            if isinstance(X, pd.DataFrame):
                X = X.loc[mask, :]  # dataframe notation
            else:
                X = X[mask, :]  # array notation
        else:
            mask = np.ones_like(y, dtype=bool)

        if y is not None:
            return X, y[mask]
        else:
            return X

    def fit(self, X, y, outlier_contam=0.4, **fit_params):
        print(f"Fitting model on {X.shape[0]} samples, {X.shape[1]} features...")
        # filter dials data
        X, y = self.filter_outliers(X, y, contamination=outlier_contam)

        self.is_trained_ = True
        return self.model.fit(X, y, **fit_params)

    def predict(self, X, confidence=0.5, outlier_contam=0.4, *args, **kwargs):
        print(f"Predicting outcome of {X.shape[0]} samples...")
        # filter dials data
        X = self.filter_outliers(X, contamination=outlier_contam)

        # get predicted probabilities of success and filter for given confidence threshold
        y_proba = self.model.predict_proba(X, *args, **kwargs)[:, 1]
        y_pred = y_proba >= confidence
        return y_pred

    def test(self, X_test, y_test, outlier_contam=0, scoring=None):
        # filter X and y
        X_test, y_test = self.filter_outliers(X_test, y_test, contamination=outlier_contam)

        # predict given samples and get metric scores
        y_pred = self.predict(X_test, outlier_contam=0)
        scoring = self.metrics if scoring is None else scoring
        score_dict = {metric: self.available_metrics_[metric](y_test, y_pred) for metric in scoring}
        return score_dict

    def summary(self):
        info = (f"Experimental Phasing prediction model for {self.data_pipeline.upper()} data, with settings:\n"
                f"> Predictor: {self.clf}\n"
                f"> Scaler: {self.scaler}\n"
                f"> Feature Selector: {self.feat_selector}\n"
                f"> Outlier Removal: {self.outlier()}\n"
                f"> Model trained: {self.is_trained_}")
        return info

    def __str__(self):
        return f"Experimental Phasing prediction model with {self.clf} predictor"

