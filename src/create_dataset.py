import os
import re
import numpy as np
import pandas as pd
import yaml
from transform import Transform
from decimal import Decimal
import argparse


def get_header(filename):
    """
    Create a header based on the file name
    :param filename: file name
    :return: header
    """
    if filename == 'bhand_nodetact_array':
        col = ['timestamp']
        # key = index start - end
        #   finger1 = 0-23
        #   finger2 = 24-48
        #   finger3 = 49-73
        #   palm = 74-98
        for finger_index in [1, 2, 3]:
            # finger_[1|2|3]_i where i = 0,1,2,...,72
            col.extend(['finger' + str(finger_index) + '_' + str(sensor_index) for sensor_index in range(0, 24)])

        # insert the header for the palm variables
        col.extend(['palm_' + str(sensor_index) for sensor_index in range(0, 24)])

        col.extend(['grasp'])

    elif filename == 'tf':
        # rotations = rw,rx,ry,rz
        # translation = tx,ty,tz
        col = ['timestamp', 'rw', 'rx', 'ry', 'rz', 'tx', 'ty', 'tz', 'grasp']

    elif filename == 'joint_states':
        # e = effort
        # p = position
        col = ['timestamp', 'e_j23', 'e_j12', 'e_j22', 'e_j32', 'e_j33', 'e_j13', 'e_j11', 'e_j21',
               'p_j23', 'p_j12', 'p_j22', 'p_j32', 'p_j33', 'p_j13', 'p_j11', 'p_j21', 'grasp']
    else:
        raise ValueError('Wrong descriptor passed to the header function')

    return col


def append_to_dataset(data_matrix, r_data, filename, n_row, n_cols):

    # check if all the values have been included
    if np.shape(r_data)[1] != n_cols:
        raise ValueError('Missing values for {0} in line {1}'.format(filename), n_row)

    # check if the dataset has been initialized
    if np.shape(data_matrix)[0] == 0:
        # create dataset
        data_matrix = np.empty(shape=(1, n_cols), dtype='object')
        # add first entry
        data_matrix[0] = np.array(r_data)
    # add row to the dataset
    else:
        data_matrix = np.vstack([data_matrix, r_data])

    return data_matrix


def convert_bhand_to_csv(yalm_file_dir, filename, s_grasp):
    """
    Read the bhand yaml file. Use the the description to create a header for the file and save its data into a csv file
    :param yalm_file_dir: directory of the yaml file
    :param filename: name of the yaml file
    :param s_grasp: 
    :return: None - create a csv file and saves the information to it
    """

    # header
    header = get_header(filename)

    yalm_object = open(yalm_file_dir)
    print 'reading {0}'.format(filename)

    # variable to keep all the data points
    dataset = np.empty(shape=(0, 0), dtype='object')
    # columns = timestamp, finger1_[0:23], finger2_[24:48], finger3_[49:73], palm_[74:98], success
    data_point = np.zeros(shape=(1, 98), dtype='object')
    # if True then record values
    val_flag = False
    # this variable is used to keep track of the number of times it look through the finger and palm check condition
    finger_palm_iter = 0

    # timestep variable
    time_step = list()

    # index used to keep track of how many points have been added to the data_point array
    # it starts at 1 because index 0 is used for the timestamp
    last_index = 1

    for row_number, line in enumerate(yalm_object):

        if ('finger' in line) or ('palm' in line):

            # this flag can only be True if all the values where inserted in the row_data variable
            if finger_palm_iter == 4:

                # insert the grasp success or unsuccessful in the last element
                data_point[0, -1] = s_grasp
                dataset = append_to_dataset(data_matrix=dataset, r_data=data_point, filename=filename,
                                            n_row=row_number, n_cols=98)

                # reset all the variables
                finger_palm_iter = 0
                data_point = np.zeros(shape=(1, 98), dtype='object')
                time_step = list()
                last_index = 0

            # turn flag True in order to read values in the next lines
            val_flag = True
            # keep count of how many times it has go through this condition
            finger_palm_iter += 1

            # create new tmp_row_data
            tmp_data_point = list()

        elif val_flag:
            # get the value in 'line'
            val = Decimal(re.search('\- (.*)\\n', line).group(1))
            # added to the list of row values
            tmp_data_point.append(val)

            if len(tmp_data_point) == 24.0:
                val_flag = False
                # insert the 24 values for either the fingers or the paml and move the index
                data_point[0, last_index:last_index + 24] = np.array(tmp_data_point)
                last_index += 24

        elif 'secs' in line:
            val = re.search('\s*secs:\s(.*)\\n', line).group(1)
            time_step.append(val)
            # there is sec and nsec
            if len(time_step) == 2:
                # switch the nsecs with the secs values in order to get the right datetime
                time_step[0], time_step[1] = time_step[1], time_step[0]
                timestep = ''.join(time_step)
                # seconds have to go before all the other fingers and palm values
                data_point[0, 0] = Decimal(timestep)

    return header, dataset


def convert_tf_to_csv(yalm_file_dir, filename, s_grasp):
    """
    Read the tf yaml file, calculate transforms for links, create a header for the file and save it to a csv
    :param yalm_file_dir: directory of the yaml file
    :param filename: name of the yaml file
    :param s_grasp: successful grasp value
    :return: None - create a csv file and saves the information to it
    """

    # header
    header = get_header(filename)

    # read file
    yalm_object = open(yalm_file_dir)
    print 'reading {0}'.format(filename)

    # variable to keep all the data points
    dataset = np.empty(shape=(0, 0), dtype='object')

    # binary flag used to know when to start collecting data
    val_flag = False
    # variable used to keep all the information for the row of the file
    data_point = np.zeros(shape=(1, 9), dtype='object')
    # keep track of initial time
    time_step = list()

    for row_index, line in enumerate(yalm_object):

        # if reading on line for link2, turn on flag in order to start collecting data
        if 'child_frame_id: staubli_rx60l_link2' in line:
            transform_data = Transform()
            val_flag = True

        elif 'secs' in line and val_flag:

            val = re.search('\s*secs:\s(.*)\\n', line).group(1)

            time_step.append(val)

            if len(time_step) == 2:
                # switch the nsecs with the secs values in order to get the right datetime
                time_step[0], time_step[1] = time_step[1], time_step[0]
                timestep = ''.join(time_step)
                # seconds have to go before all the other fingers and palm values
                data_point[0, 0] = Decimal(timestep)

        elif ('w:' in line or 'x:' in line or 'y:' in line or 'z:' in line) and val_flag:
            # extract and collect the sensor value
            val = Decimal(re.search('\s *.:\s(.*)\\n', line).group(1))
            transform_data.add(val)

            # check if all the necessary values have been added
            if transform_data.all_data_included():
                tmp_data_point = list(transform_data.get_rotation_translation())
                data_point[0, 1:-1] = np.array(tmp_data_point, dtype='object')
                data_point[0, -1] = s_grasp
                dataset = append_to_dataset(data_matrix=dataset, r_data=data_point, filename=filename,
                                            n_row=row_index, n_cols=9)

                # reset variables
                val_flag = False
                time_step = list()
                data_point = np.zeros(shape=(1, 9), dtype='object')
                transform_data = Transform()

    return header, dataset


def convert_js_to_csv(yalm_file_dir, filename, s_grasp):
    """
    Read the yaml file, obtain the joint states of all the jointsG, create a header for the file and save it as csv
    :param yalm_file_dir: directory of the yaml file
    :param filename: name of the yaml file
    :param s_grasp: successful grasp value
    :return: None - create a csv file and saves the information to it
    """
    # header
    header = get_header(filename)

    # read file
    yalm_object = open(yalm_file_dir)
    print 'reading {0}'.format(filename)

    # variable to keep all the data points
    dataset = np.empty(shape=(0, 0))

    # binary flag used to know when to start collecting data
    val_flag = False
    # keep track of initial time
    time_step = list()

    # variable used to keep all the information for the row of the file related to the effort and position of the bhand
    effort_position_data_list = list()

    # keep track of the data is corrupted
    corrupted_data = False

    for line in yalm_object:

        # if reading on line for link2, turn on flag in order to start collecting data
        if 'effort' in line or 'position' in line:

            # corrupted dataset will start with line = 'effort: []'
            if '[' in line:
                corrupted_data = True
            # the corrupted dataset will end only for the next effort flag
            elif corrupted_data and 'effort' in line:
                corrupted_data = False
                val_flag = True
            # if its corrupted and its a position line, then it will still be corrupted
            elif corrupted_data and 'position' in line:
                pass
            else:
                val_flag = True

        elif 'secs' in line and not corrupted_data:

            val = re.search('\s*secs:\s(.*)\\n', line).group(1)

            time_step.append(val)

            if len(time_step) == 2:
                # switch the nsecs with the secs values in order to get the right datetime
                time_step[0], time_step[1] = time_step[1], time_step[0]
                timestep = ''.join(time_step)

        elif val_flag:
            # extract the sensor value
            val = line.replace('- ', '').replace('\n', '')
            if len(effort_position_data_list) <= 16:
                effort_position_data_list.append(val)

                # turn the val_flag back off if all the values have been collected
                if len(effort_position_data_list) == 8 or len(effort_position_data_list) == 16:
                    val_flag = False

            else:
                raise ValueError('Wrong number of element in the effort and position data lists')

        # if both list have all the necessary values
        if len(effort_position_data_list) == 16:

            # include all the information in an array, 1 timestep, 16 sensor data and 1 grasp info
            concatenated_information = np.zeros(shape=(1, len(effort_position_data_list) + 2), dtype='object')
            concatenated_information[0, 0] = Decimal(timestep)
            concatenated_information[0, 1:-1] = effort_position_data_list
            concatenated_information[0, -1] = s_grasp

            if np.shape(dataset)[0] == 0:
                dataset = concatenated_information
            else:
                dataset = np.append(dataset, concatenated_information, axis=0)

            corrupted_data = False
            effort_position_data_list = list()
            time_step = list()

    return header, dataset


def convert_yaml_to_csv(_dir):
    """
    Takes a yaml file and saves it as csv with its success column
    :param _dir: directory of the yaml files
    :return: (None) saves the yaml as a csv 
    """

    # load and read the success.yaml file
    success_filename = os.path.join(_dir, 'success.yaml')
    stream = open(success_filename, 'r')
    successful_grasp = yaml.load_all(stream).next()['data']

    print 'grasp successful = {0}'.format(successful_grasp)

    # loop through all possible yaml file for a trial
    # for yaml_file in ['bhand_nodetact_array.yaml', 'tf.yaml', 'joint_states.yaml']:
    for yaml_file in ['tf.yaml']:
        # create the directory to the file
        yaml_file_dir = os.path.join(_dir, yaml_file)

        # check if file exists
        if not os.path.exists(yaml_file_dir):
            raise IOError('file {0} not found in dir {1}'.format(yaml_file, _dir))

        # filename
        desc = yaml_file.replace('.yaml', '')

        if 'bhand' in desc:
            header, data = convert_bhand_to_csv(yalm_file_dir=yaml_file_dir, filename=desc,
                                                s_grasp=int(successful_grasp))
        elif 'tf' in desc:
            header, data = convert_tf_to_csv(yalm_file_dir=yaml_file_dir, filename=desc, s_grasp=int(successful_grasp))
        # joint_state desc
        else:
            header, data = convert_js_to_csv(yalm_file_dir=yaml_file_dir, filename=desc, s_grasp=int(successful_grasp))

        # create ouptut csv file path
        csv_dir = yaml_file_dir.replace('yaml', 'csv')
        # convert to pandas dataframe and store it with the headers
        pd_array = pd.DataFrame(data, columns=header)
        pd_array.to_csv(csv_dir, index=False)
        print 'data {0} saved to {1}'.format(desc, csv_dir)
        exit(0)


def combine_csv(src, dst, indx):

    bhand_file_dir = os.path.join(src, 'bhand_nodetact_array.csv')
    tf_file_dir = os.path.join(src, 'tf.csv')
    joint_state_file_dir = os.path.join(src, 'joint_states.csv')

    # read the files through pandas
    bhand_file_obj = pd.read_csv(bhand_file_dir)
    tf_file_obj = pd.read_csv(tf_file_dir)
    js_file_obj = pd.read_csv(joint_state_file_dir)

    # remove the grasp column from the header
    header = list(bhand_file_obj.keys())[1:-1]
    header.extend(list(tf_file_obj.keys())[1:])

    # get the numpy values
    bhand_data = bhand_file_obj.values
    tf_data = tf_file_obj.values

    # obtain the last position for the joint 23 since this position is the hand close
    joint23_hand_position = js_file_obj['p_j23'].iloc[-1]
    # get all the values that match that position to obtain the initial dataset row where it happens
    position_rows = np.where(js_file_obj['p_j23'] == joint23_hand_position)
    # get the timestamp where it first occurs
    initial_closing_timestamp = str(js_file_obj['timestamp'].iloc[position_rows[0][0]])[:11]

    # create dict with all the tf values where key is the timestamp and value is the whole data point
    tf_data_dict = dict()

    for tf_point in tf_data:
        key = str(Decimal(tf_point[0]))[:11]
        tf_data_dict[key] = tf_point

    tf_rows, tf_cols = np.shape(tf_data)
    bhand_rows, bhand_cols = np.shape(bhand_data)
    # need to subtract both timestamp columns and one of the label columns
    total_cols = bhand_cols + tf_cols - 3
    # add a column on which to have a TRUE label referring to the data point on which to train and test the data
    # when not considering the time dependency
    total_cols += 1

    # create new dataset with the right number of columns
    # this array has bhand number of rows because it will have at most that many rows
    new_dataset = np.zeros(shape=(bhand_rows, total_cols))

    # variable use to keep the row index of the initial bhand close grip
    closing_timestamp_index = ''
    shortest_timestamp_difference = np.infty
    row_index = 0

    for bhand_point in bhand_data:
        # get the timestamp for hat point
        timestamp = str(Decimal(bhand_point[0]))[:11]
        try:
            # if that timestamp is found in the keys of the tf dictionary
            n_data = tf_data_dict[timestamp]

            if timestamp == initial_closing_timestamp:
                closing_timestamp_index = row_index

            # if the timestamp does not directly correlate, keep track of the closest timestamp
            elif not closing_timestamp_index:
                timestamp_difference = Decimal(np.abs(Decimal(timestamp) - Decimal(initial_closing_timestamp)))
                if timestamp_difference < shortest_timestamp_difference:
                    shortest_timestamp_difference = timestamp_difference
                    closest_closing_timestamp_index = row_index

            # insert the values related to that key in the new_dataset array
            new_dataset[row_index, :96] = bhand_point[1:-1]  # bhand data points
            new_dataset[row_index, 96: -1] = n_data[1:]  # tf data points

            row_index += 1

        except KeyError:
            pass

    # obtain the actual number of row with data
    fn_rows, fn_cols = np.shape(new_dataset[:row_index])
    # create a new array with an extra column that will be used to insert the trial index
    final_dataset = np.zeros(shape=(fn_rows, fn_cols + 1))
    # create a trial index array
    index_col = np.array([indx] * np.shape(final_dataset)[0])

    # insert the trial index column, data point columns and the closing timestamp column
    final_dataset[:, 0] = index_col
    final_dataset[:, 1:] = new_dataset[:row_index]

    if closing_timestamp_index:
        final_dataset[closing_timestamp_index: closing_timestamp_index + 20, -1] = 1
    else:
        final_dataset[closest_closing_timestamp_index: closest_closing_timestamp_index + 20, -1] = 1

    # check if the closing point was added
    if np.shape(np.where(final_dataset[:, -1] == 1)[0])[0] == 0:
        raise ValueError('no closing point was added')

    # change header
    final_header = ['trial_index']
    final_header.extend(header)
    final_header.extend(['closing_point'])

    if os.path.exists(dst):
        with open(dst, mode='a') as obj_file_out:
            pd_array = pd.DataFrame(final_dataset)
            pd_array.to_csv(obj_file_out, index=False, header=False)
    else:
        with open(dst, mode='w') as obj_file_out:
            pd_array = pd.DataFrame(final_dataset, columns=final_header)
            pd_array.to_csv(obj_file_out, index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Grasp Assertiveness Script')
    parser.add_argument('-d', '--directory', help='dataset directory', required=True)
    parser.add_argument('-c', '--combine', help='combine the bhand, tf and joint_state csv files', action='store_true')
    parser.add_argument('-y', '--convert', help='convert the bhand, tf and joint_state yaml files to csv',
                        action='store_true')
    args = parser.parse_args()

    # check directory exists
    if not os.path.isdir(args.directory):
        raise IOError('Directory = {0} not found.'.format(args.directory))
    else:
        data_path = args.directory

    # extract list of object(folders) in the data directory
    _, object_list, _ = os.walk(data_path).next()

    final_dataset_dir = os.path.join(data_path, 'dataset.csv')
    dataset_error = os.path.join(data_path, 'error_dataset.txt')
    annotation_file = os.path.join(data_path, 'annotation.txt')

    # remove older database.csv file
    if os.path.exists(final_dataset_dir):
        os.remove(final_dataset_dir)

    # keep track of where a trial start and ends when combining all the csv files
    trial_index = 0
    # file to know which csv are giving errors
    error_file = open(dataset_error, 'wb+')
    # file to keep track of the object, trial, index
    annotation_file_obj = open(annotation_file, 'wb+')

    for current_object in object_list:

        print 'processing {0}'.format(current_object)

        # create directory path for the current object
        object_directory = os.path.join(data_path, current_object)

        # extract list of trials(folders) in the given directory
        _, trial_folder_name_list, _ = os.walk(object_directory).next()

        for trial in trial_folder_name_list:

            try:
                print 'folder {0}'.format(trial)
                # specific trial folder = location of the yaml files
                current_trial_directory = os.path.join(object_directory, trial)

                if args.convert:
                    convert_yaml_to_csv(_dir=current_trial_directory)

                # used after converting yaml files to csv in order to merge them
                if args.combine:
                    combine_csv(src=current_trial_directory, dst=final_dataset_dir, indx=trial_index)

                msg = 'object = {0}, trial = {1}, index = {2} \n'.format(current_object, trial, trial_index)
                annotation_file_obj.write(msg)
                trial_index += 1

            except IOError:
                error_file.write(current_trial_directory + '\n')
