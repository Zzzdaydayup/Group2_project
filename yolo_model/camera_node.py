import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

class CameraNode:
    def __init__(self):
        # 初始化节点
        rospy.init_node('camera_publisher', anonymous=True)
        self.image_pub = rospy.Publisher('/camera/image_raw', Image, queue_size=10)
        self.bridge = CvBridge()
        
        # 打开摄像头
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            rospy.logerr("无法打开摄像头")
            return
        
        # 设置摄像头参数，例如分辨率和帧率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        rospy.loginfo("摄像头节点已启动")
        
    def start(self):
        rate = rospy.Rate(10)  # 设置发布频率为10Hz
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if not ret:
                rospy.logerr("无法读取摄像头帧")
                continue

            try:
                # 转换图像并发布
                msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                self.image_pub.publish(msg)
            except CvBridgeError as e:
                rospy.logerr(f"CvBridge错误: {e}")
            
            rate.sleep()
    
    def cleanup(self):
        # 释放摄像头并关闭所有窗口
        self.cap.release()
        cv2.destroyAllWindows()
        rospy.loginfo("摄像头节点已关闭")

if __name__ == '__main__':
    try:
        camera_node = CameraNode()
        camera_node.start()
    except rospy.ROSInterruptException:
        pass
    finally:
        camera_node.cleanup()
