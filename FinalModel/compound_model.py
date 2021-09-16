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

    def __init__(self, data_pipeline="dials", model_kind="MLP", model_params={},
                 feat_select_type=None, feat_select_params={"k": 7},
                 metrics=("accuracy", "f1", "mcc", "precision", "recall"),
                 outlier_kind="LOF", default_contam=None,
                 verbose=1):
        '''
        Initialise EPModel with desired configurations
        :param data_pipeline: Either `dials`, `xds`, or `3dii`. Specifies which pipeline the data was processed with.
                              This changes the default contamination and features used. xds is an alias for 3dii.
        :param model_kind: Either `MLP`, `RandomForest`, or `SVC`. Specifies which classifier to use.
        :param model_params: Dictionary specifying the specific model parameters to customise `model_kind` with.
        :param feat_select_type: If `kbest`, use SelectKBest feature selector, with parameters set by `feat_select_params`.
                                 Else use default, pre-selected set of features, depending on `data_pipeline`.
        :param feat_select_params: Dictionary specifying SelectKBest parameters. Main usage of specifying k.
                                   Defaults to {"k": 7}.
        :param metrics: Metrics to calculate during test. Can be any combination of `accuracy`, `f1`, `mcc`, `precision`, and `recall`.
                        Uses all by default.
        :param outlier_kind: Either `LOF`, `IsolationForest`, or `EllipticEnvelope`.
                             Specifies method to use when removing outliers.
                             Defaults to LOF.
        :param default_contam: Outlier contamination to use when contamination isn't specified during fit/predict/test calls.
                               Defaults to 0.4 for DIALS, and 0 for XDS
        :param verbose: Either 1 or 0. If 1, print short status messages during fitting/predicting/testing.
                        If 0, don't print these.
                        Defaults to 1.
        '''

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
            self.feat_selector = self.feat_selections_["KBest"](**feat_select_params)
            self.model.steps.insert(0, ("feat_select", self.feat_selector))
        else:
            # default, select predefined features
            trsfs = [("feat_select", "passthrough", self.feat_selections_["Default"][data_pipeline])]
            self.feat_selector = ColumnTransformer(trsfs)
            self.model.steps.insert(0, ("feat_select", self.feat_selector))

        # store settings
        self.data_pipeline = data_pipeline
        self.feat_select_type = feat_select_type
        self.metrics = metrics
        self.outlier = self.available_outliers_[outlier_kind]
        self.verbose = verbose

        # store default contamination
        if default_contam is None:
            # 40% for dials, 0% for XDS if default
            self.default_contam = 0.4 if data_pipeline == "dials" else 0
        else:
            self.default_contam = default_contam

    def filter_outliers(self, X, y=None, contamination=0.4):
        '''
        Filter dataset(s) according to given contamination. If contamination not in [0, 1], return untouched dataset(s).
        :param X: Main dataset in which to detect outliers and obtain a mask from.
                  Will always be filtered and returned if possible
        :param y: Secondary dataset to filter.
                  Will not be used for outlier detection, but will be filtered and returned whenever possible.
        :param contamination: How much of the data to consider to be outliers. Simply, fraction of data to be removed.
        :return: X if y is not given, else return X, y
        '''

        if contamination <= 0 or contamination >= 1:
            if y is not None:
                return X, y
            else:
                return X

        if self.data_pipeline == "dials":
            # only filter if dials data
            mask = self.outlier(contamination=contamination).fit_predict(X) == 1
            if self.verbose == 1:
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

    def fit(self, X, y, outlier_contam=None, **fit_params):
        '''
        Fit current model to training dataset
        :param X: Training features. 2D required.
        :param y: Training targets/labels. 1D required.
        :param outlier_contam: Contamination used when removing outliers. None for default values.
        :param fit_params: Any additional parameters to be passed to the underlying pipeline's fit method.
        :return: Returns the trained model
        '''

        if self.verbose == 1:
            print(f"Fitting model on {X.shape[0]} samples, {X.shape[1]} features...")
        # filter dials data
        contamination = self.default_contam if outlier_contam is None else outlier_contam
        X, y = self.filter_outliers(X, y, contamination=contamination)

        self.is_trained_ = True
        return self.model.fit(X, y, **fit_params)

    def predict(self, X, confidence=0.5, outlier_contam=None, *args, **kwargs):
        '''
        Predict EP outcome using given features. Model must have been previously trained.
        :param X: Feature dataset to use for prediction.
        :param confidence: Confidence threshold. Minimum certainty for model prediction decision
                           (e.g. a confidence of 80% means that the samples predicted to succeed will be
                           those 80% or more likely to succeed).
        :param outlier_contam: See fit's outlier_contam.
        :param args: Arguments to be passed to the underlying pipeline's predict_proba method.
        :param kwargs: Keyword arguments to be passed to the underlying pipeline's predict_proba method.
        :return: Returns the predictions 1D array
        '''

        if self.verbose == 1:
            print(f"Predicting outcome of {X.shape[0]} samples...")
        if self.is_trained_ is False:
            print("Model not trained yet. Please call fit() on a training dataset before attempting any predictions.")
            return None
        # filter dials data
        contamination = self.default_contam if outlier_contam is None else outlier_contam
        X = self.filter_outliers(X, contamination=contamination)

        # get predicted probabilities of success and filter for given confidence threshold
        y_proba = self.model.predict_proba(X, *args, **kwargs)[:, 1]
        y_pred = y_proba >= confidence
        return y_pred

    def test(self, X_test, y_test, confidence=0.5, outlier_contam=0, scoring=None):
        '''
        Test model on test dataset. Model must have been previously trained.
        :param X_test: Feature dataset to use for prediction.
        :param y_test: Target/label dataset to use for comparison with predictions and obtaining scores.
        :param confidence: See predict's confidence.
        :param outlier_contam: See fit's outlier_contam
        :param scoring: Scoring metrics to calculate and return.
                        Any combination from 'accuracy', 'f1', 'mcc', 'precision', 'recall'.
                        Uses all by default
        :return: Dictionary of scores names and values.
        '''

        if self.verbose == 1:
            print(f"Testing model on {X_test.shape[0]} samples")
        if self.is_trained_ is False:
            print("Model not trained yet. Please call fit() on a training dataset before attempting any predictions.")
            return None
        # filter X and y
        contamination = self.default_contam if outlier_contam is None else outlier_contam
        X_test, y_test = self.filter_outliers(X_test, y_test, contamination=contamination)

        # predict given samples and get metric scores
        y_pred = self.predict(X_test, confidence=confidence, outlier_contam=0)
        scoring = self.metrics if scoring is None else scoring
        score_dict = {metric: self.available_metrics_[metric](y_test, y_pred) for metric in scoring}
        return score_dict

    def summary(self):
        '''
        Obtain summary of classifier, scaler, feature selector, etc.
        :return: Formatted string specifying parameters/settings.
        '''

        info = (f"Experimental Phasing prediction model for {self.data_pipeline.upper()} data, with settings:\n"
                f"> Predictor: {self.clf}\n"
                f"> Scaler: {self.scaler}\n"
                f"> Feature selector: {self.feat_selector}\n"
                f"> Outlier removal: {self.outlier()}\n"
                f"> Default contamination: {self.default_contam}\n"
                f"> Model trained: {self.is_trained_}")
        return info

    def __str__(self):
        return f"Experimental Phasing prediction model with {self.clf} predictor"

    def __getitem__(self, item):
        # when queried, search in underlying feat_select-scale-classifier pipeline
        return self.model[item]
