import os
import time
import numpy as np
import yaml
import rospy
import moveit_commander
import sensor_msgs.msg
import std_msgs.msg
import subprocess
import signal
import time
# project path
program_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])


class DataCollector:

    def __init__(self):

        self.group = moveit_commander.MoveGroupCommander('StaubliArm')
        self.hand_publisher =  rospy.Publisher('/bhand_node/command', sensor_msgs.msg.JointState, queue_size=10)
        self.success_publisher =  rospy.Publisher('/success', std_msgs.msg.Bool, queue_size=10)
        self.bag_process = None
        self._current_tactile_state = np.empty(shape=[50000, 98])  # timestamp, 96 values
        self._current_joint_state = np.empty(shape=[50000, 8])  # timestamp, 6 or 7 values

        self._current_hand_pose = None  # jake #this is a pose, x,y,z, qx,qy,qz,qw
        self._object_name = None  # Jorge, cin to get name
        self._successful = False

        self._current_tactile_state_index = 0
        self._current_joint_state_index = 0

    def js_cb(self, msg):
        self._current_joint_state[self._current_joint_state_index] = msg
        self._current_joint_state_index += 1

    def tactile_state_cb(self, msg):
        self._current_tactile_state[self._current_tactile_state_index] = msg
        self._current_tactile_state_index += 1

    def close_hand(self):
        msg = sensor_msgs.msg.JointState()
        msg.header.stamp.secs = 0
        msg.header.stamp.nsecs = 0
        msg.name = ['bh_j11_joint', 'bh_j32_joint', 'bh_j12_joint', 'bh_j22_joint']
        msg.position = [0, 2.41, 2.41, 2.41]
        msg.velocity = [0.0, 0.1, 0.1, 0.1]
        msg.effort = [0, 0, 0, 0]
        self.hand_publisher.publish(msg)
        rospy.sleep(1.0)

    def lift(self):
        
        self.group.set_start_state_to_current_state()
        self.group.set_named_target('lift')

        p = self.group.plan()

        self.group.execute(p)

    def open_hand(self):
        msg = sensor_msgs.msg.JointState()
        msg.header.stamp.secs = 0
        msg.header.stamp.nsecs = 0
        msg.name = ['bh_j11_joint', 'bh_j32_joint', 'bh_j12_joint', 'bh_j22_joint']
        msg.position = [0, 0.0, 0.0, 0.0]
        msg.velocity = [0.0, 0.1, 0.1, 0.1]
        msg.effort = [0, 0, 0, 0]
        self.hand_publisher.publish(msg)
        rospy.sleep(1.0)

    def lower(self):
        self.group.set_start_state_to_current_state()
        self.group.set_named_target('pre_grasp')

        p = self.group.plan()
        self.group.execute(p)

    def ask_if_success(self):
        """
        Asks if the task was successful (1) or non-successful (0) and adds the column to the dataset
        :return: [NONE] updates self._current_joint_state, self._current_tactile_state
        """
        selected_option = raw_input('Successful grasp?[Y/N]: ').upper()
        if not (selected_option == 'Y' or selected_option == 'N'):
            selected_option = raw_input('Successful grasp?[Y/N]: ').upper()

        if selected_option == 'Y':
            self.success_publisher.publish(True)
        else:
            self.success_publisher.publish(False)

    def start_bag(self):
        self.bag_process = subprocess.Popen(['rosbag', 'record', '-a'])
        
    def end_bag(self):
        p = self.bag_process
        import psutil
        process = psutil.Process(p.pid)
        for sub_process in process.get_children(recursive=True):
            sub_process.send_signal(signal.SIGINT)
        p.wait()  # we wait for children to terminate
        
        try:
            p.terminate()
        except:
            pass
        
    def do_grasp(self):
        self.start_bag()

        self.close_hand()
        
        self.lift()
        
        self.ask_if_success()
        
        self.end_bag()

        self.open_hand()
        self.open_hand()
        
        self.lower()
        

if __name__ == '__main__':
    rospy.init_node("staubli_moveit_commander")

    dataCollector = DataCollector()
    rospy.sleep(1.0)
    dataCollector.open_hand()
    while 1:
       dataCollector.do_grasp()

    do_grasp()
    import IPython
    IPython.embed()
    



