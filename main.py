import rospy

from navigation import DijkstraPathPlanning
from hardware_control import LocomotionController
from intelligent_grasp import IntelligentGrasp  # 导入封装的智能抓取
from face_detect import FaceDetect

if __name__ == "__main__":
    # 初始化 ROS 节点
    rospy.init_node('hardware_control', log_level=rospy.DEBUG)

    # 创建抓取控制实例
    grasp_controller = IntelligentGrasp()
     # 创建路径规划实例
    path_planner = DijkstraPathPlanning()

    # 调用pathPlanner方法来获取路径和其他相关数据
    result = path_planner.pathPlanner()
    print(result)

    # 创建运动控制实例
    path_controller = LocomotionController(result)

    # 路线分三段执行，这样中间可以穿插取货，人脸识别送货等运动
    print(path_controller.pickup_path)
   
    path_controller.move_control(path_controller.pickup_path)

    path_controller.rotate()
    path_controller.stop()

    # 启动抓取功能
    grasp_controller.enter_func(None)
    grasp_controller.start_running()
    
    # 运行一段时间来测试抓取功能
    try:
        while not grasp_controller.grasp_complete:  # 等待抓取完成
            rospy.sleep(1)

        # 抓取完成后，继续执行小车移动的代码
        rospy.loginfo("Grasping complete, now moving the robot base.")

    except rospy.ROSInterruptException:
        pass
    finally:
        # 停止抓取功能
        grasp_controller.stop_running()
        grasp_controller.exit_func(None)
        grasp_controller.lift_item()
        rospy.loginfo("Shutting down"
                      )
    path_controller.rotate(90)
    path_controller.stop()


    path_controller.move_control(path_controller.destination_path)
    path_controller.stop()

    face_detect = FaceDetect()
    face_detect.set_grasp_controller(grasp_controller)
    #卸货
    grasp_controller.unload()


    #返回
    path_controller.move_control(path_controller.return_path)

    


    
