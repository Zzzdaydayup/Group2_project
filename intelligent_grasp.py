import sys
import cv2
import time
import math
import rospy
import numpy as np
from threading import RLock, Timer, Thread

from std_srvs.srv import *
from std_msgs.msg import *
from sensor_msgs.msg import Image

from sensor.msg import Led
from visual_processing.msg import Result
from visual_processing.srv import SetParam
from hiwonder_servo_msgs.msg import MultiRawIdPosDur

from armpi_pro import PID
from armpi_pro import Misc
from armpi_pro import bus_servo_control
from kinematics import ik_transform

class IntelligentGrasp:
    def __init__(self):
        self.grasp_complete = False  # Add this line
        self.lock = RLock()
        self.ik = ik_transform.ArmIK()
        
        self.x_dis = 500
        self.y_dis = 0.15
        self.img_w = 640
        self.img_h = 480
        self.centreX = 320
        self.centreY = 410
        self.offset_y = 0
        self.stable = False
        self.arm_move = False
        self.__isRunning = False
        self.position_en = False
        self.detect_color = 'None'
        self.x_pid = PID.PID(P=0.06, I=0, D=0)  # PID initialization
        self.y_pid = PID.PID(P=0.00003, I=0, D=0)

        self.range_rgb = {
            'red': (0, 0, 255),
            'blue': (255, 0, 0),
            'green': (0, 255, 0),
            'black': (0, 0, 0),
            'white': (255, 255, 255),
        }

        self.n = 0
        self.num = 0
        self.last_x = 0
        self.last_y = 0
        self.color_buf = []
        self.color_list = {0: 'None', 1: 'red', 2: 'green', 3: 'blue'}
        self.result_sub = None
        self.heartbeat_timer = None
        self.move_thread = None

        # Initialize ROS publishers and services
        self.joints_pub = rospy.Publisher('/servo_controllers/port_id_1/multi_id_pos_dur', MultiRawIdPosDur, queue_size=1)
        self.buzzer_pub = rospy.Publisher('/sensor/buzzer', Float32, queue_size=1)
        self.rgb_pub = rospy.Publisher('/sensor/rgb_led', Led, queue_size=1)

        self.enter_srv = rospy.Service('/intelligent_grasp/enter', Trigger, self.enter_func)
        self.exit_srv = rospy.Service('/intelligent_grasp/exit', Trigger, self.exit_func)
        self.running_srv = rospy.Service('/intelligent_grasp/set_running', SetBool, self.set_running)
        self.heartbeat_srv = rospy.Service('/intelligent_grasp/heartbeat', SetBool, self.heartbeat_srv_cb)

        rospy.sleep(0.5)  # Delay to ensure publishers are active

    def initMove(self, delay=True):
        with self.lock:
            target = self.ik.setPitchRanges((0, 0.15, 0.03), -180, -180, 0)
            if target:
                servo_data = target[1]
                bus_servo_control.set_servos(self.joints_pub, 1800, ((1, 200), (2, 500), (3, servo_data['servo3']), (4, servo_data['servo4']),
                                                                     (5, servo_data['servo5']), (6, servo_data['servo6'])))
        if delay:
            rospy.sleep(2)


    def unload(self):
        rospy.loginfo("Unloading items")
        with self.lock:
            # Rotate base to the left by 90 degrees (adjust the increment as necessary)
            current_base_pos = 500  # Assuming the center position is 500
            target_base_pos = current_base_pos + 360  # Rotate left by 90 degrees (assuming full range is 4096)

            # Limit the range to prevent out of bound issues
            if target_base_pos > 4096:
                target_base_pos -= 4096

            bus_servo_control.set_servos(self.joints_pub, 1000, ((6, target_base_pos),))
            rospy.sleep(5)

            # Lower the arm by moving joints 3, 4, and 5 downwards
            lower_arm_positions = {
                3: 300,  # Adjust these values to ensure the arm moves downwards
                4: 780,  # Reverse the direction for joint 4
                5: 300,
            }
        
            bus_servo_control.set_servos(self.joints_pub, 1000, (
                (3, lower_arm_positions[3]), 
                (4, lower_arm_positions[4]), 
                (5, lower_arm_positions[5]), 
                (6, target_base_pos)
            ))
            rospy.sleep(1.5)

            # Open the gripper
            bus_servo_control.set_servos(self.joints_pub, 500, ((1, 120),))
            rospy.sleep(0.5)

            # Lift the arm by moving joints 3 and 4 upwards
            lift_arm_positions = {
                3: 250,  # Adjust these values to lift the arm back up
                4: 850,
                5: 700
            }
        
            bus_servo_control.set_servos(self.joints_pub, 1000, (
                (3, lift_arm_positions[3]), 
                (4, lift_arm_positions[4]), 
                (5, lift_arm_positions[5]), 
                (6, target_base_pos)
            ))
            rospy.sleep(1.5)

            # Rotate the base back to the initial position
            bus_servo_control.set_servos(self.joints_pub, 1000, ((6, current_base_pos),))
            rospy.sleep(1.5)

            rospy.loginfo("Unload complete")

    def lift_item(self):
         # Lift the arm by moving joints 3 and 4 upwards
            lift_arm_positions = {
                3: 250,  # Adjust these values to lift the arm back up
                4: 850,
                5: 700
            }
        
            bus_servo_control.set_servos(self.joints_pub, 1000, (
                (3, lift_arm_positions[3]), 
                (4, lift_arm_positions[4]), 
                (5, lift_arm_positions[5]), 
            ))
            rospy.sleep(1.5)

    def off_rgb(self):
        led = Led()
        led.index = 0
        led.rgb.r = 0
        led.rgb.g = 0
        led.rgb.b = 0
        self.rgb_pub.publish(led)
        led.index = 1
        self.rgb_pub.publish(led)


    def init(self):
        rospy.loginfo("intelligent grasp Init")
        self.initMove()
        self.reset()


    def move(self):
        while self.__isRunning:
            if self.arm_move and self.detect_color != 'None':  # Wait for grasping
                target_color = self.detect_color  # Temporarily store target color
                self.set_rgb(target_color)  # Set RGB LED color
                rospy.sleep(0.1)
                bus_servo_control.set_servos(self.joints_pub, 500, ((1, 120),))  # Open the gripper
                rospy.sleep(0.5)
                target = self.ik.setPitchRanges((0, round(self.y_dis + self.offset_y, 4), -0.08), -180, -180, 0)  # Arm stretches downwards
                if target:
                    servo_data = target[1]
                    bus_servo_control.set_servos(self.joints_pub, 1000, ((3, servo_data['servo3']), (4, servo_data['servo4']),
                                                                         (5, servo_data['servo5']), (6, self.x_dis)))
                rospy.sleep(1.5)
                bus_servo_control.set_servos(self.joints_pub, 500, ((1, 450),))  # Close the gripper
                rospy.sleep(0.8)

                bus_servo_control.set_servos(self.joints_pub, 1500, ((1, 450), (2, 500), (3, 80), (4, 825), (5, 625), (6, 500)))  # Lift the arm
                rospy.sleep(1.5)

                # Make sure the arm holds the position and doesn't move
                self.__isRunning = False  # Stop the main loop
                self.arm_move = False     # Stop arm movement
                self.off_rgb()            # Turn off RGB LEDs

                rospy.loginfo("Grasping complete, arm lifted and holding position.")
                self.grasp_complete = True  # Indicate that grasping is complete
                return  # Exit the function to stop further execution
            else:
                rospy.sleep(0.01)

    def stop_running(self):
        rospy.loginfo("stop running intelligent grasp")
        with self.lock:
            self.__isRunning = False
            self.reset()
            if self.move_thread is not None:
                self.move_thread.join()
            # Comment out the following line to prevent reinitializing the arm position
            # self.initMove(delay=False)
            rospy.ServiceProxy('/visual_processing/set_running', SetParam)()

    def reset(self):
        with self.lock:
            self.x_dis = 500
            self.y_dis = 0.15
            self.x_pid.clear()
            self.y_pid.clear()
            self.off_rgb()
            self.arm_move = False
            self.position_en = False
            self.detect_color = 'None'
            # Do not reset arm position here to keep the grasped object held



    def run(self, msg):
        # Image processing result callback function
        center_x = msg.center_x
        center_y = msg.center_y
        color_num = msg.data
        if color_num >=2:
            color_num = 1
        if not self.position_en:  # Check if the block is stable
            dx = abs(center_x - self.last_x)
            dy = abs(center_y - self.last_y)
            self.last_x = center_x
            self.last_y = center_y
            if dx < 3 and dy < 3:
                self.n += 1
                if self.n == 10:
                    self.n = 0
                    self.position_en = True  # Stabilized
            else:
                self.n = 0

        else:
            # Block is stable, track and grasp
            if not self.arm_move and color_num != 0:
                diff_x = abs(center_x - self.centreX)
                diff_y = abs(center_y - self.centreY)
                # X-axis PID tracking
                if diff_x < 10:
                    self.x_pid.SetPoint = center_x  # Setpoint
                else:
                    self.x_pid.SetPoint = self.centreX

                self.x_pid.update(center_x)  # Current value
                dx = self.x_pid.output  # Output
                self.x_dis += int(dx)
                self.x_dis = 200 if self.x_dis < 200 else self.x_dis
                self.x_dis = 800 if self.x_dis > 800 else self.x_dis
                # Y-axis PID tracking
                if diff_y < 10:
                    self.y_pid.SetPoint = center_y  # Setpoint
                else:
                    self.y_pid.SetPoint = self.centreY

                self.y_pid.update(center_y)  # Current value
                dy = self.y_pid.output  # Output
                self.y_dis += dy
                self.y_dis = 0.12 if self.y_dis < 0.12 else self.y_dis
                self.y_dis = 0.28 if self.y_dis > 0.28 else self.y_dis

                # Arm moves to the top of the block
                target = self.ik.setPitchRanges((0, round(self.y_dis, 4), 0.03), -180, -180, 0)
                if target:
                    servo_data = target[1]
                    bus_servo_control.set_servos(self.joints_pub, 20, ((3, servo_data['servo3']), (4, servo_data['servo4']),
                                                                       (5, servo_data['servo5']), (6, self.x_dis)))

                if dx < 2 and dy < 0.003 and not self.stable:  # Wait for the arm to stabilize above the block
                    self.num += 1
                    if self.num == 10:
                        self.stable = True
                        self.num = 0
                else:
                    self.num = 0

                if self.stable:  # Multiple confirmations of the detected color
                    self.color_buf.append(color_num)
                    if len(self.color_buf) == 5:
                        mean_num = np.mean(self.color_buf)
                        if mean_num == 1.0 or mean_num == 2.0 or mean_num == 3.0:
                            self.detect_color = self.color_list[int(mean_num)]
                            self.offset_y = Misc.map(target[2], -180, -150, -0.04, 0.03)  # Set position compensation
                            self.arm_move = True  # Set the arm to grasp
                        self.color_buf = []
                        self.stable = False

    def enter_func(self, msg):
        rospy.loginfo("enter intelligent grasp")
        self.init()
        with self.lock:
            if self.result_sub is None:
                rospy.ServiceProxy('/visual_processing/enter', Trigger)()
                self.result_sub = rospy.Subscriber('/visual_processing/result', Result, self.run)

        return [True, 'enter']

    def exit_func(self, msg):
        rospy.loginfo("exit intelligent grasp")
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
            except BaseException as e:
                rospy.loginfo('%s', e)

        return [True, 'exit']

    def start_running(self):
        rospy.loginfo("start running intelligent grasp")
        with self.lock:
            self.__isRunning = True
            visual_running = rospy.ServiceProxy('/visual_processing/set_running', SetParam)
            visual_running('colors', 'rgb')
            rospy.sleep(0.1)
            # Run sub-thread
            self.move_thread = Thread(target=self.move)
            self.move_thread.setDaemon(True)
            self.move_thread.start()


    def set_running(self, msg):
        if msg.data:
            self.start_running()
        else:
            self.stop_running()

        return [True, 'set_running']

    def set_rgb(self, color):
        with self.lock:
            led = Led()
            led.index = 0
            led.rgb.r = self.range_rgb[color][2]
            led.rgb.g = self.range_rgb[color][1]
            led.rgb.b = self.range_rgb[color][0]
            self.rgb_pub.publish(led)
            rospy.sleep(0.05)
            led.index = 1
            self.rgb_pub.publish(led)
            rospy.sleep(0.05)

    def heartbeat_srv_cb(self, msg):
        if isinstance(self.heartbeat_timer, Timer):
            self.heartbeat_timer.cancel()
        if msg.data:
            self.heartbeat_timer = Timer(5, rospy.ServiceProxy('/intelligent_grasp/exit', Trigger))
            self.heartbeat_timer.start()
        rsp = SetBoolResponse()
        rsp.success = msg.data

        return rsp

