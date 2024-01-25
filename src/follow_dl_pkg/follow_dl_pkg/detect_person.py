import supervision as sv
from ultralytics import YOLO
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg        import Image
from geometry_msgs.msg      import Point
from cv_bridge              import CvBridge, CvBridgeError
import ball_tracker.process_image as proc


# Copyright 2023 Josh Newans
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node
from sensor_msgs.msg        import Image
from geometry_msgs.msg      import Point
from cv_bridge              import CvBridge, CvBridgeError
import ball_tracker.process_image as proc



class DetectPerson(Node):

    def __init__(self):
        super().__init__('detect_person')

        self.get_logger().info('Looking for the ball...')
        self.image_sub = self.create_subscription(Image,"/image_in",self.callback,rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value)
        self.image_out_pub = self.create_publisher(Image, "/image_out", 1)
        self.image_tuning_pub = self.create_publisher(Image, "/image_tuning", 1)
        self.ball_pub  = self.create_publisher(Point,"/detected_person",1)

        self.declare_parameter('tuning_mode', False)

        self.declare_parameter("x_min",0)
        self.declare_parameter("x_max",100)
        self.declare_parameter("y_min",0)
        self.declare_parameter("y_max",100)
        self.declare_parameter("h_min",0)
        self.declare_parameter("h_max",180)
        self.declare_parameter("s_min",0)
        self.declare_parameter("s_max",255)
        self.declare_parameter("v_min",0)
        self.declare_parameter("v_max",255)
        self.declare_parameter("sz_min",0)
        self.declare_parameter("sz_max",100)
        
        self.tuning_mode = self.get_parameter('tuning_mode').get_parameter_value().bool_value
        self.tuning_params = {
            'x_min': self.get_parameter('x_min').get_parameter_value().integer_value,
            'x_max': self.get_parameter('x_max').get_parameter_value().integer_value,
            'y_min': self.get_parameter('y_min').get_parameter_value().integer_value,
            'y_max': self.get_parameter('y_max').get_parameter_value().integer_value,
            'h_min': self.get_parameter('h_min').get_parameter_value().integer_value,
            'h_max': self.get_parameter('h_max').get_parameter_value().integer_value,
            's_min': self.get_parameter('s_min').get_parameter_value().integer_value,
            's_max': self.get_parameter('s_max').get_parameter_value().integer_value,
            'v_min': self.get_parameter('v_min').get_parameter_value().integer_value,
            'v_max': self.get_parameter('v_max').get_parameter_value().integer_value,
            'sz_min': self.get_parameter('sz_min').get_parameter_value().integer_value,
            'sz_max': self.get_parameter('sz_max').get_parameter_value().integer_value
        }

        self.model = YOLO('./runs/detect/train3/weights/best.pt') #####
        self.bounding_box_annotator = sv.BoxAnnotator(thickness=1, text_thickness=1, text_scale=0.5)

        self.bridge = CvBridge()

        if(self.tuning_mode):
            proc.create_tuning_window(self.tuning_params)

    
    def normalise_keypoint(self, cv_image, detections):
        rows = float(cv_image.shape[0])
        cols = float(cv_image.shape[1])
        center_x    = 0.5 * cols
        center_y    = 0.5 * rows

        kp_x = int(np.mean([detections.xyxy[0, 0], detections.xyxy[0, 2]]))
        kp_y = int(np.mean([detections.xyxy[0, 1], detections.xyxy[0, 3]]))

        obj_area = (detections.xyxy[0, 2] - detections.xyxy[0, 0]) * (detections.xyxy[0, 3] - detections.xyxy[0, 1])
        total_area = rows * cols

        x = (kp_x[0] - center_x) / (center_x)
        y = (kp_y[1] - center_y) / (center_y)
        z = obj_area / total_area
 
        return x, y, z

    
    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        try:            
            result = self.model(cv_image, agnostic_nms=True)[0]
            detections = sv.Detections.from_yolov8(result)
            labels = [f"{self.model.model.names[class_id]} {confidence:0.2f}" for _, confidence, class_id, _ in detections]
            out_image = self.box_annotator.annotate(scene=cv_image, detections=detections, labels=labels)

            img_to_pub = self.bridge.cv2_to_imgmsg(out_image, "bgr8")
            img_to_pub.header = data.header
            self.image_out_pub.publish(img_to_pub)

            # img_to_pub = self.bridge.cv2_to_imgmsg(tuning_image, "bgr8")
            img_to_pub.header = data.header
            self.image_tuning_pub.publish(img_to_pub)

            point_out = Point()

            x_norm, y_norm, z_norm = self.normalise_keypoint(cv_image, detections)

            point_out.x = x_norm
            point_out.y = y_norm
            point_out.z = z_norm

        except CvBridgeError as e:
            print(e)  


def main(args=None):

    rclpy.init(args=args)

    detect_person = DetectPerson()
    while rclpy.ok():
        rclpy.spin_once(detect_person)
        proc.wait_on_gui()

    detect_person.destroy_node()
    rclpy.shutdown()

