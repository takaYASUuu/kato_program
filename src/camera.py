#!/usr/bin/env python

import numpy as np
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

def camera_callback(msg):
    try:
        bridge = CvBridge()
        cv_array = bridge.imgmsg_to_cv2(msg)
        #img = cv2.cvtColor(cv_array, cv2.COLOR_BGR2GRAY)
        cv2.imshow('image', cv_array)
        cv2.waitKey(1)
        rospy.loginfo(cv_array)
 
    except Exception as err:
        rospy.logerr(err)

def start_node():
    rospy.init_node('camera')
    rospy.loginfo('camera node started')
    rospy.Subscriber("/gmsl/image_0", Image, camera_callback)
    rospy.spin()

if __name__ == '__main__':
    try:
        start_node()
    except rospy.ROSInterruptException:
        pass