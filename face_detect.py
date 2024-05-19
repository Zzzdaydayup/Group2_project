#!/usr/bin/python3
# coding=utf8
# Date:2022/05/30
import sys
import cv2
import math
import rospy
import numpy as np
from threading import RLock, Timer, Thread

from std_srvs.srv import *
from std_msgs.msg import *
from sensor_msgs.msg import Image
from visual_processing.msg import Result
from visual_processing.srv import SetParam

class FaceDetect:
    def __init__(self):
        self.lock = RLock()
        self.__isRunning = False
        self.result_sub = None
        self.heartbeat_timer = None
        self.grasp_controller = None

        self.img_w = 640
        self.center_x = 0
        self.center_y = 0

        self.enter_srv = rospy.Service('/face_detect/enter', Trigger, self.enter_func)
        self.exit_srv = rospy.Service('/face_detect/exit', Trigger, self.exit_func)
        self.running_srv = rospy.Service('/face_detect/set_running', SetBool, self.set_running)
        self.heartbeat_srv = rospy.Service('/face_detect/heartbeat', SetBool, self.heartbeat_srv_cb)

    def set_grasp_controller(self, grasp_controller):
        self.grasp_controller = grasp_controller

    def reset(self):
        with self.lock:
            self.center_x = 0
            self.center_y = 0

    def init(self):
        rospy.loginfo("face detect Init")
        self.reset()

    def run(self, msg):
        self.center_x = msg.center_x
        self.center_y = msg.center_y

        # 判断人脸是不是在画面中间
        if abs(self.center_x - self.img_w / 2) < 100:
            print("Face detected, move to unload item")
            if self.grasp_controller:
                self.grasp_controller.unload()

    def enter_func(self, msg):
        rospy.loginfo("enter face detect")
        with self.lock:
            self.init()
            if self.result_sub is None:
                rospy.ServiceProxy('/visual_processing/enter', Trigger)()
                self.result_sub = rospy.Subscriber('/visual_processing/result', Result, self.run)
        return [True, 'enter']

    def exit_func(self, msg):
        rospy.loginfo("exit face detect")
        with self.lock:
            rospy.ServiceProxy('/visual_processing/exit', Trigger)()
            self.__isRunning = False
            self.reset()
            try:
                if self.result_sub is not None:
                    self.result_sub.unregister()
                    self.result_sub = None
                if self.heartbeat_timer is not None:
                    self.heartbeat_timer.cancel()
                    self.heartbeat_timer = None
            except:
                pass
        return [True, 'exit']

    def start_running(self):
        rospy.loginfo("start running face detect")
        with self.lock:
            self.__isRunning = True

    def stop_running(self):
        rospy.loginfo("stop running face detect")
        with self.lock:
            self.reset()
            self.__isRunning = False

    def set_running(self, msg):
        rospy.loginfo("%s", msg)
        if msg.data:
            visual_running = rospy.ServiceProxy('/visual_processing/set_running', SetParam)
            visual_running('face', '')
            self.start_running()
        else:
            self.stop_running()
        return [True, 'set_running']

    def heartbeat_srv_cb(self, msg):
        if isinstance(self.heartbeat_timer, Timer):
            self.heartbeat_timer.cancel()
        if msg.data:
            self.heartbeat_timer = Timer(5, rospy.ServiceProxy('/face_detect/exit', Trigger))
            self.heartbeat_timer.start()
        rsp = SetBoolResponse()
        rsp.success = msg.data
        return rsp

if __name__ == '__main__':
    rospy.init_node('face_detect', log_level=rospy.DEBUG)
    face_detect = FaceDetect()

    debug = False
    if debug:
        face_detect.enter_func(1)
        face_detect.start_running()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")
    finally:
        cv2.destroyAllWindows()
