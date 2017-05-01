# coding: utf-8

import rospy
import sensor_msgs.msg

node = rospy.init_node("me")

pub = rospy.Publisher("/bhand_node/command", sensor_msgs.msg.JointState, queue_size=10)

msg = sensor_msgs.msg.JointState()
msg.header.stamp.secs=0
msg.header.stamp.nsecs=0
msg.name = ['bh_j11_joint', 'bh_j32_joint', 'bh_j12_joint', 'bh_j22_joint']
msg.position = [0.5, 0.5, 1.0, 1.5]
msg.velocity = [0.1, 0.1, 0.1, 0.1]
msg.effort = [0, 0, 0, 0]


pub.publish(msg)

