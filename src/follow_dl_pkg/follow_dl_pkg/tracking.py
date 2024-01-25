import cv2
import torch
import numpy as np
from ultralytics import YOLO
import supervision as sv

from cv_bridge import CvBridge
import rclpy as rp
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from ament_index_python.packages import get_package_share_directory


class PiCamSubscriber(Node):
    def __init__(self):
        super().__init__('picam_subscriber')
        self.picam_subscriber = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.image_callback,
            10
        )
        self.bridge = CvBridge() 
        self.model = YOLO('yolov8n.pt')
        self.box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=2, text_scale=1)
        self.image = np.empty(shape=[1])
        # self.picam_subscriber  # prevent unused variable warning

    def image_callback(self, data):
        self.image = self.bridge.compressed_imgmsg_to_cv2(data)
        yolo_result = self.model(self.image, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(yolo_result)
        self.image = self.box_annotator.annotate(scene=self.image, detections=detections)
        
        cv2.imshow('picam_img', self.image)
        # file_name = time.strftime("%Y%m%d-%H%M%S") + ".jpg"
        # path = os.path.join(get_package_share_directory('follow_dl_pkg'), 'img_saved', file_name)
        # cv2.imwrite(path, self.image)

        key = cv2.waitKey(1)