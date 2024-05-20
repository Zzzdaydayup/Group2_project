Welcome to group2's robotic project!


1.Hardware Requirements:
1)Hiwonder Armpi pro with Raspberry Pi

2)Yahboom Autopilot Track Map UV printing canvas (2.8m*3.2m)

(btw: You'll most likely need a screwdriver)

2.Environment Requirement:

1)Python==3.8
2)ROS system

Instruction:

1.clone repository

2.run the main.py using command "python3 main.py"

3.in deafult mode, the task will be randomly generated by navigation
(you may customize your task by setting desitination and pick-up place manually in navigation.py)

4.any modification for map can be done in config.py (you may use the mapping.py to extract your map info)

4.traffic signal detection can be customised in hardware_control.py

5.face detection module currently is only used at the desitination place due to the performance restriction

To-do:

in future version, we'll focus on the improvement of detection model and logic with higher accuracy and wider applicability.


