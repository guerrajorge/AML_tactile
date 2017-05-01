import argparse
import os
import pandas as pd
import numpy as np
from sklearn import svm
from hmmlearn import hmm
from sklearn.linear_model import LogisticRegression
from collections import Counter
import random
import logging
from utils.logger import logger_initialization


def most_common(lst):
    m_frequent = Counter(lst).most_common(2)
    # if out of the two most frequent labels, the first one is
    # larger than the second one i.e. different, return the first one (largest frequency)
    if len(m_frequent) > 1:
        if m_frequent[0][1] != m_frequent[1][1]:
            return m_frequent[0][0]
        # if there are two equally frequent labels, randomly return one
        else:
            rv = random.randint(0, 1)
            return m_frequent[rv][0]
    else:
        return m_frequent[0][0]


def reshape_predictions(predictions, labels):

    dataset = np.zeros(shape=(np.shape(predictions)[0], 2))
    dataset[:, 0] = predictions
    dataset[:, 1] = labels
    window_size = 150
    step = 30
    chunks = sliding_window(sequence=dataset, window_size=window_size, step=step)

    new_dataset = list()
    new_labels = list()

    for segmented_data in chunks:
        # obtain labels
        segmented_labels = list(segmented_data[:, -1])
        # get the most common label
        new_labels.append(most_common(segmented_labels))
        # separate the labels from the dataset
        n_dataset = segmented_data[:, :np.shape(segmented_data)[1] - 1]
        average_of_probabilities = np.mean(n_dataset, axis=0)

        # add values and labels to dataframes
        new_dataset.append(average_of_probabilities)

    return np.array(new_dataset), np.array(new_labels)


def sliding_window(sequence, window_size, step=1):
    """
    Returns a generator that will iterate through
    the defined chunks of input sequence. Input sequence
    must be sliceable.
    """

    # Verify the inputs
    if not isinstance(type(window_size), type(0)) and isinstance(type(step), type(0)):
        raise Exception("**ERROR** type(window_size) and type(step) must be int.")
    if step > window_size:
        raise Exception("**ERROR** step must not be larger than window_size.")
    if window_size > len(sequence):
        raise Exception("**ERROR** window_size must not be larger than sequence length.")

    # Pre-compute number of chunks to emit
    number_of_chunks = ((len(sequence) - window_size) / step) + 1

    # Do the work
    for i in range(0, number_of_chunks * step, step):
        yield sequence[i: i + window_size]


def train_test_split(array, test_size=0.25, random_state=0):
    """
    Split time-series arrays or matrices into random train and test subsets
    
    Parameters
    ----------
    *array : sequence of indexables with same length / shape[0]

    test_size : float. Should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the test split. 

    random_state : int or RandomState
        Pseudo-random number generator state used for random sampling.
    """
    np.random.seed(random_state)

    # format the indices information
    unique_indices = set(array.index)
    possible_indices = np.array(list(unique_indices))
    n_indices = int(len(possible_indices) * test_size)

    # obtain testing dataset
    test_indices = np.sort(np.random.choice(possible_indices, n_indices, replace=False))
    test_dataset = array.ix[test_indices]

    # obtain training dataset
    train_indices = unique_indices - set(test_indices)
    train_dataset = array.ix[train_indices]

    training_subset_indices = list()
    for train_index in list(train_indices):
        array_loc = np.where(train_dataset.index == train_index)
        training_subset_indices.append(array_loc[0][-1] - array_loc[0][0])

    testing_subset_indices = list()
    for test_index in list(test_indices):
        array_loc = np.where(test_dataset.index == test_index)
        testing_subset_indices.append(array_loc[0][-1] - array_loc[0][0])

    training_closing_indices = np.where(train_dataset['closing_point'] == 1)
    testing_closing_indices = np.where(test_dataset['closing_point'] == 1)

    return train_dataset.drop(['grasp', 'closing_point'], axis=1), \
        test_dataset.drop(['grasp', 'closing_point'], axis=1), train_dataset['grasp'], \
        test_dataset['grasp'], training_subset_indices, testing_subset_indices, training_closing_indices[0], \
        testing_closing_indices[0]


def main():
    parser = argparse.ArgumentParser(description='Grasp Assertiveness Script')
    parser.add_argument('-d', '--directory', help='dataset directory', required=True)
    parser.add_argument("-l", "--log", dest="logLevel", choices=['DEBUG', 'INFO', 'ERROR'],
                        help="Set the logging level")
    args = parser.parse_args()

    # check directory exists
    if not os.path.isdir(args.directory):
        msg = 'Directory = {0} not found.'.format(args.directory)
        raise IOError(msg)
    else:
        working_dir = args.directory

    logger_initialization(logger_dir=working_dir, parser=parser)

    logging.getLogger('time.info').info('Running the Grasp Assertiveness Script')

    try:
        dataset_dir = os.path.join(working_dir, 'dataset.csv')
        dataset = pd.read_csv(dataset_dir)
    except IOError:
        msg = 'Could not find \'dataset.csv\' in directory {0}'.format(working_dir)
        logging.getLogger('info').error(msg)
        raise IOError(msg)

    # index the dataset based on the trial index
    indexed_dataset = dataset.set_index(keys=['trial_index'])

    random_state = 7
    testing_size = 0.10
    logging.getLogger('info').info('Obtaining training and testing datasets')
    x_train, x_test, y_train, y_test, train_indx_list, test_indx_list, train_cl_indx, test_cl_indx = train_test_split(
                                                                                            indexed_dataset,
                                                                                            test_size=testing_size,
                                                                                            random_state=random_state)

    msg = 'training and testing parameters:\n\t\ttesting size = {0}, random state = {1}'.format(testing_size,
                                                                                                random_state)
    logging.getLogger('tab.info').info(msg)

    # converting the index to int (to remove decimal) and the convert them to string to be able to print them all
    x_train_index = [str(int(indx)) for indx in set(x_train.index)]
    x_test_index = [str(int(indx)) for indx in set(x_test.index)]
    training_indices = ','.join(x_train_index)
    testing_indices = ','.join(x_test_index)
    msg = 'training indices:\n\t\t' + training_indices + '\n\ttesting indices:\n\t\t' + testing_indices
    logging.getLogger('tab.info.line').info(msg)

    training_dataset = x_train.values[train_cl_indx, :]
    training_labels = y_train.values[train_cl_indx]
    testing_dataset = x_test.values[test_cl_indx, :]
    testing_labels = y_test.values[test_cl_indx]

    logging.getLogger('info').info('Running SVM')
    # training and testing on time-independent dataset
    clf = svm.SVC()
    clf.fit(training_dataset, training_labels)
    svm_score = clf.score(testing_dataset, testing_labels)

    msg = 'SVM score = {0}'.format(svm_score)
    logging.getLogger('tab.info').info(msg)

    logging.getLogger('info').info('Running HMM')
    n_pos_components = [2, 5, 7, 15]
    cov_types = ['diag', 'tied', 'spherical']
    n_iterations = [5, 10, 20, 50]
    for nc in n_pos_components:
        for cov in cov_types:
            for _iter in n_iterations:

                try:
                    msg = 'running HMM with the following parameters'
                    logging.getLogger('time.info').info(msg)
                    msg = 'number of states = {0}, type of covariance = {1}, number of iterations = {2}'.format(nc,
                                                                                                                cov,
                                                                                                                _iter)
                    logging.getLogger('tab.info').info(msg)
                    # training and testing on time-dependent dataset
                    hmm_model = hmm.GaussianHMM(n_components=nc, random_state=random_state, covariance_type=cov,
                                                n_iter=_iter)
                    hmm_model.fit(x_train, lengths=train_indx_list)

                    # training hmm and logistic regression
                    hmm_training_predictions = hmm_model.predict(x_train, lengths=train_indx_list)

                    hmm_training_predictions_reshaped, labels_processed = reshape_predictions(
                        predictions=hmm_training_predictions, labels=y_train)

                    msg = 'running Logistic Regression'
                    logging.getLogger('tab.time.info').info(msg)

                    # mapping hmm labels to true labels
                    logistic_regression_model = LogisticRegression()
                    logistic_regression_model.fit(X=hmm_training_predictions_reshaped, y=labels_processed)

                    # predictions on testing dataset
                    hmm_testing_predictions = hmm_model.predict(x_test, lengths=test_indx_list)
                    hmm_testing_prediction_reshaped, testing_labels_processed = reshape_predictions(
                        predictions=hmm_testing_predictions, labels=y_test)
                    time_score = logistic_regression_model.score(X=hmm_testing_prediction_reshaped,
                                                                 y=testing_labels_processed)
                    msg = 'HMM-Logistic Regression score = {0}'.format(time_score)
                    logging.getLogger('tab.time.info').info(msg)

                except ValueError as error_message:
                    msg = 'Error while processing the following parameters ' \
                          '\n\t\tnumber of states = {0}, type of covariance = {1}, number of iterations = {2}'.format(
                            nc, cov, _iter)
                    logging.getLogger('tab.info').error(msg)
                    msg = 'error message = {0}'.format(error_message)
                    logging.getLogger('tab.tab.info').error(msg)
                    pass

                msg = 'finished running HMM'
                logging.getLogger('time.info').info(msg)

    logging.getLogger('time.info').info('Finished running the Grasp Assertiveness Script')

if __name__ == '__main__':
    main()
