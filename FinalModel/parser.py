import argparse


cmd_parser = argparse.ArgumentParser(prog="EP Model",
                                     description="Command-Line interface for quick usage of "
                                                 "a custom EP Predictive Model.",
                                     epilog="This is still an initial interface, and remains largely incomplete. "
                                            "Enjoy!")

# Add main arguments to fit, predict, test and show summary


# add model fitting option
cmd_parser.add_argument("-fit", action="store", nargs=2, metavar=("X_TRAIN", "Y_TRAIN"),
                        help="Fit EP model on csv feature dataset found on X_TRAIN, using targets from Y_TRAIN,"
                             "and store model on --model_loc or --output.")

# add model predicting option
cmd_parser.add_argument("-predict", action="store", nargs=1, metavar="X_PRED",
                        help="Use already fitted EP model to predict EP outcome "
                             "using the features found on X_PRED, storing the result on --output.")

# add model testing option
cmd_parser.add_argument("-test", action="store", nargs=2, metavar=("X_TEST", "Y_TEST"),
                        help="Test already fitted EP model "
                             "on csv data file found on X_TEST, "
                             "using the targets on Y_TEST, "
                             "storing results on --output csv file.")

# add summary option
cmd_parser.add_argument("-summary", action="store_true",
                        help="Load EP model and list settings summary. Store in --output if specified.")


## Add options to specify model and storage location

# add option to specify location of model to use when testing/predicting, and where to store when fitting
model_default_loc = "EPModel.pkl"
cmd_parser.add_argument("--model_loc", action="store", default=model_default_loc,
                        help=f"Custom EP Model location. "
                             f"Specify location of where to fetch model when predicting/testing/summarising, "
                             f"and where to store when fitting. Disregarded when fitting if --output is specified. "
                             f"Defaults to {model_default_loc}.")

# add option to specify output location when testing/predicting/summarising
test_out_default, predict_out_default, summary_out_default = "test_results.csv", "predictions.csv", None
cmd_parser.add_argument("--output", action="store", default="default",
                        help="Specify output location when fitting/testing/predicting/summarising. "
                             "None results in no storing. "
                             f"Defaults to fit: {model_default_loc}, test: {test_out_default}, "
                             f"predict: {predict_out_default}, summary: None")



## Add optional fitting arguments

# add option to choose between dials and xds
available_pipelines = ["dials", "xds", "3dii"]
cmd_parser.add_argument("--pipeline", action="store", choices=available_pipelines, default="dials",
                        help="Set whether data is from DIALS (default) or XDS (aka 3DII). Only used when fitting.")


# add option to set location of classifier_params csv file
cmd_parser.add_argument("--classifier_params", action="store", default="classifier_params.txt", metavar="FILENAME",
                        help="Indicate location of text file indicating the model parameters to use, "
                             "for more complex models. Only used when fitting.")

# add option to choose feature selection process
cmd_parser.add_argument("--kbest", action="store", type=int, default=0, metavar="K",
                        help="Whether to use the default features for each pipeline (kbest = 0), "
                             "or to take the k best features (kbest = k). Only used when fitting.")

# add option specifying which outlier removal model to use
cmd_parser.add_argument("--outlier_model", action="store", default="LOF",
                        help="What outlier removal model to use. Currently support LocalOutlierFactor (LOF), "
                             "IsolationForest (IF), and EllipticEnvelope (EE). Defaults to LOF. Only used when fitting")

# add option specifying the default contamination to use
cmd_parser.add_argument("--default_contam", action="store", type=float, default=None,
                        help="The default contamination to be used when filtering for outliers in this model. "
                             "Defaults to 0.4 for DIALS and 0 for XDS. Only used when fitting.")


args = cmd_parser.parse_args()

# print(vars(args))

# CREATE FUNCTIONS TO BE CALLED


def fit_store(dataloc, targetloc, modeloc):
    '''
    Function to fit EP model on csv data found on dataloc, and then store said model in storeloc
    :param dataloc: Data features location in csv format
    :param targetloc: Data targets/labels location in csv format
    :param modeloc: EP Model location
    :return: Fitted EP model
    '''

    # first import needed modules
    from compound_model import EPModel
    from pandas import read_csv
    from joblib import dump
    from ast import literal_eval

    # try to fetch model params
    try:
        params = read_csv(args.classifier_params, squeeze=True, header=None, delimiter=":", index_col=0, names=["Features"])
        print("Custom model parameters found")
        if "classifier_kind" in params.index:
            # read model kind and parameters
            classifier_kind = params["classifier_kind"]
            params = params.drop("classifier_kind")
        else:
            classifier_kind = "MLP"  # use MLP as default

        # convert parameters dtypes into float if possible
        for ix, param in params.items():

            # convert hidden layer sizes to tuple of ints
            if ix == "hidden_layer_sizes":
                params[ix] = literal_eval(param)

            # try to convert every other to float
            else:
                try:
                    params[ix] = float(param)
                except ValueError:
                    continue

        print(f"Using {classifier_kind} model with:\n{params}\n")
        params = params.to_dict()

    except FileNotFoundError:
        # use defaults, very simple and overfitting models
        print("No custom model parameters found, using default parameters\n"
              "Consider specifying the model parameters to be used, either through --classifier_params "
              "or through adding a classifier_params.txt file in the same directory as this file\n")
        params = {}
        classifier_kind = "MLP"

    # use SelectKBest if k > 0, use default feature set otherwise
    feat_select_type = "kbest" if args.kbest > 0 else "default"
    feat_select_params = {"k": args.kbest}

    # choose outlier removal method/model
    outlier = args.outlier_model.lower()
    if outlier == "if" or outlier == "isolationforest":
        out_mdl = "IsolationForest"
    elif outlier == "ee" or outlier == "ellipticenvelope":
        out_mdl = "EllipticEnvelope"
    else:
        # default to Local Outlier Factor
        out_mdl = "LOF"

    # choose default contamination
    def_contam = args.default_contam
    if isinstance(def_contam, int):
        if def_contam < 0 or def_contam > 1:
            print(f"Default contamination of {def_contam} outside of [0, 1] range. Using default..")
            def_contam = None

    # read csv files
    df = read_csv(dataloc, index_col=0, header=0)
    targets = read_csv(targetloc, index_col=0, squeeze=True)

    # create and fit EP Model
    mdl = EPModel(data_pipeline=args.pipeline, classifier_kind=classifier_kind, classifier_params=params,
                  feat_select_type=feat_select_type, feat_select_params=feat_select_params,
                  outlier_kind=out_mdl, default_contam=def_contam)
    mdl.fit(df, targets)

    if modeloc is not None:
        # store model on given location if not None
        dump(mdl, modeloc)

    return mdl


def predict(dataloc, modeloc, storeloc):
    '''
    Function to predict EP outcome using STORELOC model and DATALOC features
    :param dataloc: Location of CSV data
    :param modeloc: Location of stored EP model
    :param storeloc: Location of where to store predictions as CSV file
    :return: Predictions
    '''

    # import needed modules
    from pandas import read_csv
    from joblib import load

    # read csv file and model
    df = read_csv(dataloc, index_col=0, header=0)
    mdl = load(modeloc)

    # quit program if model not already trained
    if mdl.is_trained_ is False:
        print("Model not trained. Please fit the model before attempting a prediction.")
        import sys
        sys.exit()

    # continue to predictions if fitted/trained
    else:
        from numpy import savetxt

        # predict
        pred = mdl.predict(df)

        # store as csv file if storeloc is not none
        if storeloc is not None:
            savetxt(storeloc, pred, delimiter=",")

        return pred


def test_model(dataloc, targetloc, modeloc, storeloc):
    '''
    Function to test already fitted model on csv dataset
    :param dataloc: Location of features' CSV file
    :param targetloc: Location of targets' CSV file
    :param modeloc: Location of EP Model
    :param storeloc: Location of test result storage, can be None for no storage
    :return: Dict of scores
    '''

    # first import needed modules
    from pandas import read_csv
    from joblib import load
    from pandas import Series

    # read csv files
    df = read_csv(dataloc, index_col=0, header=0)
    targets = read_csv(targetloc, index_col=0, squeeze=True)

    # load EP model
    mdl = load(modeloc)
    scores = mdl.test(df, targets)

    if storeloc is not None:
        # store scores if storeloc is not None
        scores_srs = Series(scores)
        scores_srs.to_csv(storeloc)

    return scores


def show_summary(modeloc, storeloc=None):
    '''
    Function to show EP model summary given a filename
    :param filename: EP Model location
    :return: summary string
    '''
    from joblib import load

    # first load model
    mdl = load(filename=modeloc)
    txt = mdl.summary()

    if storeloc is not None:
        # store summary if storeloc is not None
        with open(storeloc, "r") as fhandle:
            fhandle.write(txt)

    return txt


# CREATE FUNCTION CALLABLES LOGIC


if args.fit is not None:

    # use model_loc if output not specified, else use output
    out = args.model_loc if args.output == "default" else args.output

    # don't store if specified output is None, else store where specified
    out = None if out.lower() == "none" else out

    # fit and store if desired
    dataloc, targetloc = args.fit
    mdl = fit_store(dataloc=dataloc, targetloc=targetloc, modeloc=out)
    if out is not None:
        print(f"Stored results in {out}")

    if args.summary is not False:
        print()
        print(mdl.summary())

elif args.test is not None:
    from pandas import Series

    # specify storage location, if desired
    if args.output == "default":
        # default storage location
        out = test_out_default
    elif args.output.lower() == "none":
        # no storage location
        out = None
    else:
        # use specified location
        out = args.output

    # test and get scores
    dataloc, targetloc = args.test
    scores = test_model(dataloc=dataloc, targetloc=targetloc, modeloc=args.model_loc, storeloc=out)
    print(Series(scores))
    if out is not None:
        print(f"Stored results in {out}")

elif args.predict is not None:

    # specify storage location, if desired
    if args.output == "default":
        # default storage location
        out = predict_out_default
    elif args.output.lower() == "none":
        # no storage location
        out = None
    else:
        # use specified location
        out = args.output

    dataloc, = args.predict
    predict(dataloc=dataloc, modeloc=args.model_loc, storeloc=out)
    if out is not None:
        print(f"\nStored results in {out}")

elif args.summary is not False:

    # specify storage location, if desired
    if args.output == "default" or args.output.lower() == "none":
        # default to no storage
        out = None
    else:
        # use specified location
        out = args.output

    print(show_summary(args.model_loc))

else:
    print("Nothing found...")
