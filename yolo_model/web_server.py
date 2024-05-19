from flask import Flask, jsonify
from flask_cors import CORS
import rospy
from std_msgs.msg import String

app = Flask(__name__)
CORS(app)

detection_result = ""

def detection_callback(data):
    global detection_result
    detection_result = data.data

rospy.init_node('web_server', anonymous=True)
rospy.Subscriber('/detection_result', String, detection_callback)

@app.route('/detection', methods=['GET'])
def get_detection():
    global detection_result
    return jsonify(detection_result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
