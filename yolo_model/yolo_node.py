class YOLOv5Node:
    def __init__(self):
        rospy.init_node('yolov5_node', anonymous=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt').to(self.device)
        rospy.loginfo("YOLOv5模型加载完毕")
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback)
        self.result_pub = rospy.Publisher('/detection_result', String, queue_size=10)
        self.red_light_detected = False
        cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
        rospy.loginfo("YOLOv5节点已启动")

    def image_callback(self, data):
        try:
            frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge错误: {e}")
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(frame)
        annotated_frame = np.squeeze(results.render())
        cv2.imshow("Detection", cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
        
        detection_str = results.pandas().xyxy[0].to_json(orient="records")
        self.result_pub.publish(detection_str)
        
        self.red_light_detected = self.detect_red_light(results)

    def detect_red_light(self, results):
        for det in results.pandas().xyxy[0].itertuples():
            if det.name == "red_light" and det.confidence > 0.5:
                rospy.loginfo("检测到红灯")
                return True
        return False

    def is_red_light_detected(self):
        return self.red_light_detected

    def cleanup(self):
        cv2.destroyAllWindows()
        rospy.loginfo("YOLOv5节点已关闭")
