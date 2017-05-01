import quaternion
import numpy as np


class Transform:

    def __init__(self):
        self.cfid_list = list()

        # creating 6 child_frame_id elements and placing them in a list variable
        # 6 elements = link1->link2,link2->link3,...,link6->link7
        for i in range(0, 3):
            rot_array = np.zeros(shape=(4, 1))
            trans_array = np.zeros(shape=(3, 1))
            self.cfid_list.append(dict([('rotation', rot_array), ('translation', trans_array)]))

        self.current_element = 0
        self.rotation_translation = 'rotation'
        self.index = 0
        self.a_data_included = False
        self.rotation = 0
        self.translation = 0

    def get_rotation_translation(self):
        # create homogeneous matrix
        # homogeneous_matrix = np.zeros(shape=(4, 4))
        # homogeneous_matrix[:3, :3] = quaternion.as_rotation_matrix(quaternion.as_quat_array(self.rotation))
        # homogeneous_matrix[0:3, 3] = self.translation.T
        # homogeneous_matrix[3, 3] = 1
        # # invert it to obtain the new translation and rotation
        # inverse_homogeneous_matrix = np.linalg.inv(homogeneous_matrix)
        # new_translation = inverse_homogeneous_matrix[:-1, 3]
        # rotation = inverse_homogeneous_matrix[:3, :3]
        # new_rotation = quaternion.as_float_array(quaternion.from_rotation_matrix(rotation))[0]
        # return np.append(new_rotation, new_translation)
        return np.append(self.rotation, self.translation.T)

    def add(self, val):
        """
        include the value in the respective object parameters and calculate their rotation and translations
        :param val: sensor data
        :return: (None)
        """
        # add value to the right element
        self.cfid_list[self.current_element][self.rotation_translation][self.index] = val

        # there are 4 coordinates for rotation
        if self.index == 3:
            self.rotation_translation = 'translation'
            self.index = 0

        # there are 3 coordinates for translation
        elif self.index == 2 and self.rotation_translation == 'translation':
            self.rotation_translation = 'rotation'
            self.index = 0
            self.current_element += 1

        else:
            self.index += 1

        if self.current_element == 3:
            self.a_data_included = True

            n_rotation_matrix = np.empty(shape=(0, 0))
            self.translation = np.empty(shape=(0, 0))

            for data_point in self.cfid_list:
                # convert the rotation quaternion matrix to a 3x3 rotation matrix
                tmp_rotation_matrix = \
                    quaternion.as_rotation_matrix(quaternion.as_quat_array(data_point['rotation'].T)).reshape(3, 3)

                # check if first iteration
                if n_rotation_matrix.shape == (0, 0):
                    n_rotation_matrix = tmp_rotation_matrix
                else:
                    n_rotation_matrix = np.dot(n_rotation_matrix, tmp_rotation_matrix)

                tmp_translation_matrix = np.dot(n_rotation_matrix, data_point['translation'])

                # check if first iteration
                if not self.translation.shape == (0, 0):
                    self.translation += tmp_translation_matrix
                else:
                    self.translation = tmp_translation_matrix

            # convert it back to quaternion
            # To convert an N-dimensional array of quaternions to an Nx4 array of floats
            self.rotation = quaternion.as_float_array(quaternion.from_rotation_matrix(n_rotation_matrix))

    def all_data_included(self):
            return self.a_data_included
