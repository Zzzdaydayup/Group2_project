from config import NEIGHBOR_PATHS, PARK_NODE, FORBIDDEN_NODE, TURN_MAPPING
import sys
import rospy
from chassis_control.msg import *
start = True

class LocomotionController:
   
    def __init__(self, result):
        rospy.on_shutdown(self.stop)
        # Mecanum wheel chassis control
        self.set_velocity = rospy.Publisher('/chassis_control/set_velocity', SetVelocity, queue_size=1)
        rospy.sleep(10)
        # Save the contents of the result dictionary to instance variables
        self.result = result
        self.path = result.get('path', [])
        self.neighbor_paths = NEIGHBOR_PATHS
        # Dynamically update the path dictionary
        self.update_path()

        # Define path segments: pickup path, delivery path, and return path
        self.pickup_path, self.destination_path, self.return_path = self.split_path()
        
        # Initialize the car's direction
        self.current_direction = 'E'
     
    # Handle shutdown
    def stop(self):
        global start
        start = False
        print('Shutting down...')
        self.set_velocity.publish(0, 0, 0)  # Publish chassis control message to stop moving
        print('Shutdown complete')

    # Dynamically update path dictionary
    def update_path(self):
        # Extract information for pickup point 0 and destination 12
        pickup_info = self.result['pickup_point']
        destination_info = self.result['destination']

        # Update path information for pickup point 0
        self.neighbor_paths[(pickup_info[0], 0)] = (self.result['dist_0_1'], self.neighbor_paths[(pickup_info[0], pickup_info[1])][1])
        print("Path dictionary dynamically updated - Pickup point info:", (pickup_info[0], 0), self.neighbor_paths[(pickup_info[0], 0)])
        self.neighbor_paths[(0, pickup_info[1])] = (self.result['dist_0_2'], self.neighbor_paths[(pickup_info[0], pickup_info[1])][1])
        print("Path dictionary dynamically updated - Pickup point info:", (0, pickup_info[1]), self.neighbor_paths[(0, pickup_info[1])])

        # Update path information for destination 12
        self.neighbor_paths[(destination_info[0], 12)] = (self.result['dist_12_1'], self.neighbor_paths[(destination_info[0], destination_info[1])][1])
        print("Path dictionary dynamically updated - Destination info:", (destination_info[0], 12), self.neighbor_paths[(destination_info[0], 12)])
        self.neighbor_paths[(12, destination_info[1])] = (self.result['dist_12_2'], self.neighbor_paths[(destination_info[0], destination_info[1])][1])
        print("Path dictionary dynamically updated - Destination info:", (12, destination_info[1]), self.neighbor_paths[(12, destination_info[1])])

    # Path segmentation function: pickup - delivery - return
    def split_path(self):
        # Get the indices for the pickup point and destination from the result dictionary
        pickup_index = self.path.index(0)         
        destination_index = self.path.index(12)  
        
        # Pickup path: from the starting point to the pickup point
        pickup_path = self.path[:pickup_index+1]
        print("Pickup path:", pickup_path)

        # Delivery path: from the pickup point to the destination
        destination_path = self.path[pickup_index:destination_index+1]
        print("Delivery path:", destination_path)

        # Return path: from the destination back to the starting point
        return_path = self.path[destination_index:]
        print("Return path:", return_path)
        return pickup_path, destination_path, return_path 
    
    # Direction control function
    def turn_to_direction(self, target_direction):
        if self.current_direction != target_direction:
            turn_angle = TURN_MAPPING.get((self.current_direction, target_direction))
            print(f"Need to turn {turn_angle} degrees")
            # Call ROS rotate method, positive angle for clockwise rotation
            self.rotate(turn_angle)
            self.current_direction = target_direction  # Update current direction

    # Car rotation function
    def rotate(self, angle):
        angular_speed = -0.3 if angle > 0 else 0.3  # Negative speed for clockwise rotation, corresponding to angle>0
        time_duration = abs(angle / 24.62)
        self.set_velocity.publish(0, 90, angular_speed)
        time_duration = round(time_duration, 5)
        rospy.sleep(time_duration)
        print(f"Car turning for {time_duration} seconds")
        # Speed and time should be adjusted during testing
    
    # Car movement function
    def move_straight(self, distance):
        # Set linear speed to a positive value to move forward
        linear_speed = 8  # Assumed linear speed, time and speed to be tested
        # Publish speed message to start moving straight
        self.set_velocity.publish(60, 90, 0)
        # Calculate the movement time, distance / speed gives the time required to move (seconds)
        time_duration = distance / linear_speed
        time_duration = round(time_duration, 1)
        print(f"Car moving straight for {time_duration} seconds")
        rospy.sleep(time_duration)
    
    # Path execution function
    def move_to_next_point(self, current_point, next_point):
        distance, target_direction = self.neighbor_paths.get((current_point, next_point))
        print(f"Straight distance: {distance} Target direction: {target_direction}")
        # Handle entering and exiting the parking lot separately
        if current_point == PARK_NODE and next_point == FORBIDDEN_NODE:
            self.park_in_or_out('out', distance)
        elif current_point == FORBIDDEN_NODE and next_point == PARK_NODE:
            self.park_in_or_out('in', distance)
        else:
            # First, check if the direction is correct
            self.turn_to_direction(target_direction)
            # Then execute straight movement
            self.move_straight(distance)

    # Entering and exiting the parking lot function
    def park_in_or_out(self, flag, distance):
        # Exiting the parking lot, initial direction is 'E'
        if flag == 'in':
            # Reset direction to 'E' first
            self.turn_to_direction('E')
            linear_speed = 8  # Assumed linear speed, time and speed to be tested
            self.set_velocity.publish(60, 180, 0)  # 180 degrees, equivalent to the car moving left out of the parking lot
            time_duration = distance / linear_speed
            time_duration = round(time_duration, 1)
            rospy.sleep(time_duration)
        # Parking into the lot
        elif flag == 'out':
            linear_speed = 8 
            self.set_velocity.publish(60, 0, 0)  # 0 degrees, equivalent to the car moving right into the parking lot
            time_duration = distance / linear_speed
            time_duration = round(time_duration, 1)
            rospy.sleep(time_duration)

    def stop(self):
        self.set_velocity.publish(0,0,0)
    # Control execution function

    #update-need : more logic for detection red-light
    def check_for_red_light(self):
        rospy.loginfo("start decttion")
        
        def image_callback(data):
            try:
                frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
            except CvBridgeError as e:
                rospy.logerr(f"CvBridge error: {e}")
                return

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.model(frame)
            annotated_frame = np.squeeze(results.render())
            cv2.imshow("Detection", cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
            
            for det in results.pandas().xyxy[0].itertuples():
                if det.name == "red_light" and det.confidence > 0.5:
                    rospy.loginfo("red-light!stop!")
                    self.stop()
                    break
        
        #ros
        image_sub = rospy.Subscriber('/camera/image_raw', Image, image_callback)
        rospy.sleep(2)
        image_sub.unregister()
        
        rospy.loginfo("红灯检测完成")


    # Control path executing
    def move_control(self, path):
        for i in range(len(path)-1):
            print(f"Currently executing movement from {path[i]} to {path[i+1]}")
            self.move_to_next_point(path[i], path[i+1])
