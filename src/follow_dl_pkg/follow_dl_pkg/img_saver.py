import cv2
import numpy as np
import time
import os

from cv_bridge import CvBridge
import rclpy as rp
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from ament_index_python.packages import get_package_share_directory

bridge = CvBridge() 

class PiCamSubscriber(Node):
    def __init__(self):
        super().__init__('picam_subscriber')
        self.picam_subscriber = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.image_callback,
            10
        )
        self.image = np.empty(shape=[1])
        

    def image_callback(self, data):
        self.image = bridge.compressed_imgmsg_to_cv2(data)
        cv2.imshow('picam_img', self.image)
        file_name = time.strftime("%Y%m%d-%H%M%S") + ".jpg"
        path = os.path.join(get_package_share_directory('follow_dl_pkg'), 'img_saved', file_name)
        cv2.imshow(self.image)
        cv2.imwrite(path, self.image)

        key = cv2.waitKey(10)


class CropSubscriber(Node):
    def __init__(self):
        super().__init__('picam_subscriber')
        self.picam_subscriber = self.create_subscription(
            CompressedImage,
            '/crop_img',
            self.image_callback,
            10
        )

        self.capture_subscriber = self.create_subscription(
            String,
            '/capturing',
            self.capture_callback,
            10
        )

        self.capture_mode = False
        
        self.image = np.empty(shape=[1])

    
    def image_callback(self, data):
        print(self.capture_mode)
        if self.capture_mode == True:
            self.image = bridge.compressed_imgmsg_to_cv2(data)
            cv2.imshow('picam_img', self.image)
            file_name = time.strftime("%Y%m%d-%H%M%S") + ".jpg"
            path = os.path.join(get_package_share_directory('follow_dl_pkg'), 'img_saved', 'train', 'one_person', file_name)
            cv2.imwrite(path, self.image)

            key = cv2.waitKey(5)

    
    def capture_callback(self, msg):
        if msg.data == 'capture_start':
            self.capture_mode = True
        else:
            self.capture_mode = False

        


def main(args=None):
    rp.init(args=args)
    
    crop_subscriber = CropSubscriber()

    try :
        rp.spin(crop_subscriber)
    except KeyboardInterrupt :
        crop_subscriber.get_logger().info('Stopped by Keyboard')
    finally :
        crop_subscriber.destroy_node()
        rp.shutdown()    


if __name__ == '__main__':
    main()











