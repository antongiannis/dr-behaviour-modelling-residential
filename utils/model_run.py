import numpy as np
seed_value = 423                                          # Set random seed for reproducible results
np.random.seed(seed_value)                                # Set the `numpy` pseudo-random generator fixed
import os
import random
import errno
from pathlib import Path
from tqdm.autonotebook import tqdm, trange
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import statsmodels.formula.api as smf
from catboost import CatBoostRegressor, Pool
# from catboost.utils import get_gpu_device_count
# noinspection PyUnresolvedReferences
from category_encoders.cat_boost import CatBoostEncoder          # More user-friendly categorical encoding
# noinspection PyUnresolvedReferences
from category_encoders.glmm import GLMMEncoder
import tensorflow as tf
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
from tensorflow.keras import layers
from tensorflow import keras
from kerastuner import HyperModel                                # Hyper-parameter tuning of keras models
from kerastuner.tuners import RandomSearch
import json


# Set seed for reproducible results
os.environ['PYTHONHASHSEED'] = str(seed_value)            # Set the `PYTHONHASHSEED` environment variable fixed
random.seed(seed_value)                                   # Set the `python` built-in pseudo-random generator fixed
tf.random.set_seed(seed_value)                            # Set the `tensorflow` pseudo-random generator fixed


__version__ = '0.1.0'
__author__ = u'Ioannis Antonopoulos'


def get_parser():
    """
    Creates a new argument parser for running various ML models. It is needed to provide the relative or absolute
    path of the directory where the data is located. The default number of iterations is 5.
    """
    parser = argparse.ArgumentParser('Run various ML algorithms')
    version = '%(prog)s ' + __version__
    parser.add_argument('dir_path', type=str,
                        help='Relative path of the directory where data is located')
    parser.add_argument('--x_name', type=str, default='X_data.csv',
                        help='Name of predictors\' csv file')
    parser.add_argument('--y_name', type=str, default='y_data.csv',
                        help='Name of target variable\'s csv file')
    parser.add_argument('--iterNum', '-N', type=int, default=10,
                        help='Number of iterations to run the algorithms')
    parser.add_argument('--version', '-v', action='version', version=version)

    return parser


def get_notebook_path():
    """
    A helper function to get the notebook parent path for our specific case.
    """
    return Path.cwd()


# noinspection PyUnboundLocalVariable
def load_split_data(dir_path, x_name="X_data.csv", y_name="y_data.csv"):
    """
    Loads the predictors and target's data and split them to train, evaluation and test set.

    Parameters
    ----------
    dir_path: str
        Relative or absolute path of the data directory.
    x_name: str, optional
        The csv file name for predictors.
    y_name: str, optional
        The csv file name for the target variable.

    Returns
    -------
    A dictionary of tuples for the train, evaluation, and test set.
    """

    notebook_path = get_notebook_path()

    # Error handling for path
    try:
        x_path = notebook_path / dir_path / x_name
        y_path = notebook_path / dir_path / y_name
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    # Load data in pandas frames
    x = pd.read_csv(x_path)
    y = pd.read_csv(y_path).iloc[:, 0]                          # Needs to be as series so it will have (nrows,) shape

    # Encode objects to category columns
    nominal_features = x.select_dtypes('object').columns.to_list()
    x[nominal_features] = x[nominal_features].apply(lambda x: x.astype('category'))

    # Split data to train, val and test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.15)

    out_dict = {'train_data': (x_train, y_train),
                'eval_data': (x_val, y_val),
                'test_data': (x_test, y_test)}

    return out_dict


def data_encode(data, encoder='CatBoostEncoder'):
    """
    Encodes the data using the `category_encoders` library. Currently only `CatBoostEncoder` and `GLMMEncoder`
    are supported.

    Parameters
    ----------
    data: dict
        Dictionary of data including train, eval, and test.
    encoder: str, optional
        The name of the encoder.

    Returns
    -------
    A dictionary of tuples for the encoded train, evaluation, and test set.
    """
    # Load data
    x_train, y_train = data.get('train_data')
    x_val, y_val = data.get('eval_data')
    x_test, y_test = data['test_data']

    # Check that the encoder argument is in the accepted args list
    enc_args_list = ['CatBoostEncoder', 'GLMMEncoder']
    if encoder not in enc_args_list:
        raise ValueError('Encoder argument is not supported')

    # Numerical (no transformation at this point)
    num_features = x_train.select_dtypes(include=['float64', 'int64']).columns.to_list()
    x_train_enc = x_train[num_features].copy()
    x_val_enc = x_val[num_features].copy()
    x_test_enc = x_test[num_features].copy()
    
    # Nominal features
    cat_features = x_train.select_dtypes(['object', 'category', 'string']).columns.to_list()
    enc_ord = eval(encoder + '(cols=cat_features)')
    x_train_enc[cat_features] = enc_ord.fit_transform(x_train, y_train)[cat_features].copy()
    x_val_enc[cat_features] = enc_ord.transform(x_val)[cat_features].copy()
    x_test_enc[cat_features] = enc_ord.transform(x_test)[cat_features].copy()

    out_dict_enc = {'train_data': (x_train_enc, y_train),
                    'eval_data': (x_val_enc, y_val),
                    'test_data': (x_test_enc, y_test)}

    return out_dict_enc


def mean_absolute_percentage_error(y_true, y_pred,
                                   sample_weight=None,
                                   multioutput='uniform_average'):
    """Mean absolute percentage error regression loss.
    Note here that we do not represent the output as a percentage in range
    [0, 100]. Instead, we represent it in range [0, 1/eps]. Code from sklearn version 0.24
    """
    epsilon = np.finfo(np.float64).eps
    mape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), epsilon)
    output_errors = np.average(mape,
                               weights=sample_weight, axis=0)
    if isinstance(multioutput, str):
        if multioutput == 'raw_values':
            return output_errors
        elif multioutput == 'uniform_average':
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)


def generate_score_metrics(model, x, y):
    """
    Generate various score metrics for various algorithms.

    Parameters
    ----------
    model: model
        The trained model.
    x: numpy.array
        The predictors data to be used for the error metrics. Could be train, eval, or test data.
    y: numpy.array
        The target variable to be used for the error metrics. Could be train, eval, or test data.

    Returns
    -------
    A dictionary of various error metrics
    """
    y = np.array(y)
    # Calculate the test statistics
    y_hat = model.predict(x)
    rmse = np.sqrt(mean_squared_error(y, y_hat))
    mae = mean_absolute_error(y, y_hat)
    mape = mean_absolute_percentage_error(y, y_hat)
    r2 = r2_score(y, y_hat)

    # Output dict
    metric_dict = {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2}

    return metric_dict


# The linear model
def lm(data, remove_constant=True):
    """
    Run a linear model using statsmodels library

    Parameters
    ----------
    data: dict
        Dictionary of data including train, eval, and test.
    remove_constant: bool, optional
        Whether on not to remove the constant term.

    Returns
    -------
    A dictionary of tuples for the train, evaluation, and test set
    """

    # Get partitioned data
    x_train, y_train = data.get('train_data')
    x_test, y_test = data['test_data']

    # Create the formula R style
    constant_term = "" + "-1" * remove_constant
    all_predictors = "+".join(x_train.columns)
    formula = "median_response~" + all_predictors + constant_term

    # Create the model
    linear_model = smf.ols(formula=formula,
                           data=x_train.merge(y_train, left_index=True, right_index=True))
    res = linear_model.fit()

    return generate_score_metrics(res, x_test, y_test)


def gbr(data, **kwargs):
    """
    Train a gradient boosting regressor based on CatBoost library.

    Parameters
    ----------
    data: dict
        Dictionary of data including train, eval, and test.
    kwargs: dictionary
        A dictionary of hyper-parameters to be passed on to the Catboost algorithm.

    Returns
    -------
    A dictionary of tuples for the train, evaluation, and test set
    """

    # Get partitioned data
    x_train, y_train = data.get('train_data')
    x_val, y_val = data['eval_data']
    x_test, y_test = data['test_data']

    # Default model parameters
    default_params = {
        'iterations': 500,
        'learning_rate': 0.1,
        'eval_metric': 'RMSE',
        'logging_level': 'Silent',
        'task_type': 'CPU',                       # Small dataset so it is faster than GPU
        'boosting_type': 'Ordered',
        'use_best_model': True                    # For choosing the best model based on validation set.
    }

    # Update and add parameters if needed
    params = {**default_params, **kwargs}

    # Find categorical columns
    try:
        cat_features = x_train.select_dtypes(['object', 'category', 'string']).columns.to_list()
    except ValueError as error:
        raise error

    # Create pools
    train_pool = Pool(x_train, y_train, cat_features=cat_features)
    validate_pool = Pool(x_val, y_val, cat_features=cat_features)

    # Create the model and fit base model
    gbr = CatBoostRegressor(**params)
    gbr.fit(train_pool, eval_set=validate_pool)

    return generate_score_metrics(gbr, x_test, y_test)


def rf(data):
    """
    Train a Random forest model and return the error metrics.
    """
    # Load encoded data
    enc_data = data_encode(data, encoder='CatBoostEncoder')
    x_train_enc, y_train = enc_data.get('train_data')
    x_test_enc, y_test = enc_data.get('test_data')

    # Fit the model
    rfr = RandomForestRegressor()
    rfr.fit(x_train_enc, y_train)

    return generate_score_metrics(rfr, x_test_enc, y_test)


class MyHyperModel(HyperModel):
    """
    Class that overwrites the build method of the `kerastuner.engine.hypermodel.HyperModel` class.
    """

    def __init__(self, num_output, nun_features):
        super().__init__()
        self.num_output = num_output
        self.nun_features = nun_features

    def build(self, hp):
        # Create the NN architecture
        inputs = keras.Input(shape=(self.nun_features,), name="input_layer")
        x = layers.Dense(units=hp.Int('units',
                                      min_value=10,
                                      max_value=100,
                                      step=10),
                         activation="relu",
                         name="dense_1")(inputs)
        x = layers.Dense(units=hp.Int('units',
                                      min_value=10,
                                      max_value=100,
                                      step=10),
                         activation="relu",
                         name="dense_2")(x)
        outputs = layers.Dense(self.num_output,
                               activation="linear",
                               name="predictions")(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Training configuration
        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Choice('learning_rate',
                          values=[1e-2, 1e-3, 1e-4])),
            loss=keras.losses.MeanSquaredError(),
            metrics=['mae', 'mape'])

        return model


def get_best_nn(data, num_output=1, **tuner_kw):
    """
    Find the "best" model based on `MyHyperModel` class.

    Parameters
    ----------
    data: numpy.array or similar
        The train and validation data to be used by the hyper parameter tuner.
    num_output: int, optional
        The number of outputs for our NN. 1 default for regression.
    tuner_kw: dictionary
        A dictionary of parameters to be  `RandomSearch` tuner.

    Returns
    -------
    The trained model instance with the "optimised" parameters.
    """
    # Load encoded data
    enc_data = data_encode(data, encoder='CatBoostEncoder')
    x_train_enc, y_train = enc_data.get('train_data')
    x_val_enc, y_val = enc_data.get('test_data')

    # Create an instance of the `MyHyperModel` class
    hyper_model = MyHyperModel(num_output=num_output, nun_features=int(x_train_enc.shape[1]))

    # Default tuner params
    default_tuner_params = {'objective': 'val_loss',
                            'max_trials': 10,
                            'directory': 'keras_tuner_output',                   # Directory for logs, checkpoints, etc
                            'project_name': 'sgsc'}       # Default is utils/keras_tuner_output

    # Update tuner params
    tuner_params = {**default_tuner_params, **tuner_kw}

    # Initialise tuner and run it
    tuner = RandomSearch(hyper_model, **tuner_params)
    # Check about seed!! We need to define it? or does it use numpy's by default?
    tuner.search(x_train_enc, y_train,
                 epochs=5,                                              # Default number of epochs
                 validation_data=(x_val_enc, y_val),
                 verbose=0)

    # Get best model
    best_hp = tuner.get_best_hyperparameters()[0]
    best_model = tuner.hypermodel.build(best_hp)

    return best_model, best_hp


def fit_dense_nn(best_model, data, **kwargs):
    """
    Retrieves the best model from the Keras tuner and does inference on a data partition.

    Parameters
    ----------
    best_model: model
        The keras model.
    data: dict
        Dictionary of data including train, eval, and test.
    kwargs: dictionary
        A dictionary of parameters to be passed to the `fit()` method of the keras model.

    Returns
    -------
    A dictionary of tuples for the train, evaluation, and test set
    """
    # Load encoded data
    enc_data = data_encode(data, encoder='CatBoostEncoder')
    x_train_enc, y_train = enc_data.get('train_data')
    x_val_enc, y_val = enc_data.get('eval_data')

    # Default training parameters
    default_train_params = {'batch_size': 32,
                            'epochs': 50,
                            'verbose': 0,
                            'validation_data': (x_val_enc, y_val)
                            }

    # Update tuner params
    fit_params = {**default_train_params, **kwargs}

    # Fit data to model
    best_model.fit(x_train_enc, y_train, **fit_params)

    return best_model


def dense_nn(best_model, data, **kwargs):
    fitted_model = fit_dense_nn(best_model, data,  **kwargs)
    enc_data = data_encode(data=data, encoder='CatBoostEncoder')
    x_test_enc, y_test = enc_data.get('test_data')

    return generate_score_metrics(fitted_model, x_test_enc, y_test)


if __name__ == '__main__':
    # Get the arguments
    args = get_parser().parse_args()

    # Get the "optimised" Keras model once
    data = load_split_data(dir_path=args.dir_path, x_name=args.x_name, y_name=args.y_name)
    best_nn, best_hp = get_best_nn(data=data, num_output=1)
    print('Hyper parameter tuning for NN completed successfully')

    # Train a Keras model and save it for permutation feature importance use
    model_NN = fit_dense_nn(best_nn, data)
    model_NN.save('keras_saved_model')
    print('Saved Keras model')

    # Initialise the overall metric dict
    final_metric_dict = {}                                    # Results dictionary for all iterations

    model_list = ['lm', 'gbr', 'rf', 'dense_nn']              # List of models to run
    mod_args_list = ['(data, remove_constant=False)',         # List of parameters for each function
                     '(data)',
                     '(data)',
                     '(best_nn, data)'
                     ]

    # Outer loop per iteration
    for iteration in trange(args.iterNum, desc='Iteration loop'):
        # Get training and test data
        data = load_split_data(dir_path=args.dir_path, x_name=args.x_name, y_name=args.y_name)

        iter_metric_dict = {}                                              # Dictionary of all models per iteration
        pbar = tqdm(zip(model_list, mod_args_list),                        # tqdm Instantiation outside of the loop
                    desc='Model loop', total=len(model_list), leave=False)
        # Inner loop per model
        for model, params_str in pbar:
            # Run the models
            model_res = eval(model + params_str)

            # Update the results for each model
            iter_metric_dict.update({model: model_res})

        # Update overall results dict
        dict_key = 'iter_' + str(iteration+1)
        final_metric_dict.update({dict_key: iter_metric_dict})

    # Write the results to a file
    notebook_path = get_notebook_path()
    try:
        json_path = notebook_path / 'data' / 'result.json'
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    # noinspection PyUnboundLocalVariable
    with open(json_path, 'w') as fp:
        json.dump(final_metric_dict, fp)

print("All models have been trained successfully")
