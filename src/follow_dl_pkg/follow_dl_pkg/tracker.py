import rclpy
import cv2
import math
import numpy as np
from ultralytics import YOLO
import supervision as sv

from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
from rclpy.qos import qos_profile_sensor_data


class Tracking(Node):
    def __init__(self):
        super().__init__('edi_tracking')

        self.bridge = CvBridge()
        self.twist = Twist()
        
        self.img_sub = self.create_subscription(CompressedImage, "/camera/image_raw/compressed", self.callback, 10)
        self.mode_sub = self.create_subscription(String, "/follow", self.follow_callback, 10)
        self.img_publisher = self.create_publisher(CompressedImage, '/yolo_video', qos_profile_sensor_data)
        self.crop_publisher = self.create_publisher(CompressedImage, '/crop_img', 10)
        # self.img_sub = self.create_subscription(CompressedImage, "cctv_video", self.callback, qos_profile_sensor_data)
        self.twist_pub = self.create_publisher(Twist, "/base_controller/cmd_vel_unstamped", 10)

        self.box_annotator = sv.BoxAnnotator(thickness=1, text_thickness=1, text_scale=0.5)

        self.bridge = CvBridge()

        self.model = YOLO("/home/haneol/pinkbot/src/follow_dl_pkg/follow_dl_pkg/best.pt")

        self.out_image = None
        self.crosshair_length = 20  # You can adjust this for a smaller or larger crosshair
        
        self.object_center = 0
        self.pre_center = 0

        self.i_error_a = 0
        self.d_error_a = 0
        self.pre_center = 0

        self.follow_mode = False
        self.crop_img = None


    def normalise_keypoint(self, cv_image, detections):
        rows = float(cv_image.shape[0])
        cols = float(cv_image.shape[1])
        center_x    = 0.5 * cols
        center_y    = 0.5 * rows

        kp_x = int(np.mean([detections.xyxy[0, 0], detections.xyxy[0, 2]]))
        # kp_y = int(np.mean([detections.xyxy[0, 1], detections.xyxy[0, 3]]))

        obj_area = (detections.xyxy[0, 2] - detections.xyxy[0, 0]) * (detections.xyxy[0, 3] - detections.xyxy[0, 1])
        total_area = rows * cols

        x = (kp_x - center_x) / (center_x)
        # y = (kp_y[1] - center_y) / (center_y)
        z = obj_area / total_area
 
        return x, z
    
    
    def show_img(self, out_image):
        height, width = out_image.shape[:2]
        center_x, center_y = width // 2, height // 2
        
        cv2.line(out_image, (center_x, center_y - self.crosshair_length), (center_x, center_y + self.crosshair_length), (0, 0, 255), 1)
        cv2.line(out_image, (center_x - self.crosshair_length, center_y), (center_x + self.crosshair_length, center_y), (0, 0, 255), 1)
        
        # if detected:
        #     cv2.putText(out_image, str(self.object_center), (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 1, cv2.LINE_AA)
        #     cv2.putText(out_image, str(z_norm), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow('Window Title', out_image)
        if cv2.waitKey(10) == ord('q'):
            cv2.destroyAllWindows()
    
    def follow_callback(self, msg):
        if msg.data == "follow":
            self.follow_mode = True
        else:
            self.follow_mode = False
    
    def get_cropped_img(self, image, xyxy):
        x_start, x_end = int(xyxy[0]), int(xyxy[2])
        y_start, y_end = int(xyxy[1]), int(xyxy[3])
        cropped = image[y_start:y_end, x_start:x_end]
        
        return cropped
    
    def callback(self, msg):
        print(self.follow_mode)
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg)

        except CvBridgeError as e:
            print(e)

        try:            
            result = self.model(cv_image, agnostic_nms=True)[0]
            detections = sv.Detections.from_yolov8(result)
            detections = detections[detections.confidence > 0.85]

            if 0 in detections.class_id:
                cropped = self.get_cropped_img(cv_image, detections.xyxy[0])
                self.crop_img = self.bridge.cv2_to_compressed_imgmsg(cropped)
                labels = [f"{self.model.model.names[class_id]} {confidence:0.2f}" for _, confidence, class_id, _ in detections]
                self.out_image = self.box_annotator.annotate(scene=cv_image, detections=detections, labels=labels)

                if self.follow_mode == True:
                    x_norm, z_norm = self.normalise_keypoint(cv_image, detections)
                    print("detected area:", z_norm)                
                    
                    self.object_center = x_norm
                    if self.object_center > 0.1:
                        angular = -0.5
                    elif self.object_center < -0.1:
                        angular = 0.5
                    else:
                        angular = 0.0
                    print("angular:", angular)

                    if z_norm <= 0.25:
                        linear_x = 1.5
                    elif z_norm > 0.25 and z_norm <=0.35:
                        linear_x = 0.0
                    elif z_norm > 0.35:
                        linear_x = -1.5

                    self.publish_twist(linear_x, angular)

                self.show_img(self.out_image)
            
            else:
                # if self.pre_center < 0 :
                #     self.publish_twist(0.2, -1.0)

                # elif self.pre_center > 0:
                #     self.publish_twist(0.2, 1.0)
                
                # elif self.pre_center == 0:
                #     self.publish_twist(0.0, 0.0)

                self.show_img(cv_image)

            cvt_img = self.bridge.cv2_to_compressed_imgmsg(cv_image)
            self.img_publisher.publish(cvt_img)
            if self.crop_img:
                self.crop_publisher.publish(self.crop_img)
            
        except CvBridgeError as e:
            print(e)  


    def get_controls(self, x,  Kp_a, Ki_a, Kd_a):
        p_error_a = x
        self.i_error_a += p_error_a
        curr_d_error_a = p_error_a - self.d_error_a
        angular = Kp_a*p_error_a + Ki_a*self.i_error_a + Kd_a*curr_d_error_a
        
        if angular >= 4.0:
            angular = 4.0

        elif angular < -4.0:
            angular = -4.0
        
        return angular


    def publish_twist(self, linear_x, angular_z):
        self.twist.linear.x = linear_x
        self.twist.angular.z = angular_z
        self.twist_pub.publish(self.twist)

        
        
def main(args=None):
    rclpy.init(args=args)

    tracking = Tracking()
    
    rclpy.spin(tracking)
    
    cv2.destroyAllWindows()
    
    tracking.destroy_node()
    
    rclpy.shutdown()


if __name__ == "__main__":
    main()