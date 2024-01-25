import rclpy
import cv2
import math
import numpy as np
import os
from ultralytics import YOLO
import supervision as sv
import torchvision.models as models
import torchvision.transforms as transforms
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PIL import Image
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
from rclpy.qos import qos_profile_sensor_data
from ament_index_python.packages import get_package_share_directory



class MobileNetV3FeatureExtractor:
    def __init__(self):
        self.model = models.mobilenet_v3_large(pretrained=True) # MobileNetV3 모델 로드
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1])) # 마지막 분류 레이어 제거
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.anchor = None  # anchor 변수 초기화

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    
    def calculate_average_feature_vector(self, dataset_path, batch_size=32):
        """ 데이터셋의 평균 feature vector를 계산하는 함수 """
        dataset = ImageFolder(root=dataset_path, transform=self.transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Feature vector 계산
        features = []
        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(self.device)
                output = self.model(inputs)
                features.append(output)

        # 평균 feature vector 계산
        self.anchor = torch.mean(torch.cat(features, dim=0), dim=0)
    
    
    def preprocess_image(self, opencv_image):
        """ OpenCV 이미지를 전처리하는 함수 """
        image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        
        return self.transform(image)
    
    
    def find_most_similar(self, opencv_images):
        """ OpenCV 이미지 리스트를 입력으로 받아 self.anchor와 가장 유사한 이미지의 인덱스를 반환하는 함수 """
        preprocessed_images = [self.preprocess_image(image) for image in opencv_images]
        inputs = torch.stack(preprocessed_images).to(self.device)

        with torch.no_grad():
            features = self.model(inputs)

        anchor_normalized = F.normalize(self.anchor.view(1, -1), p=2, dim=1)
        features_normalized = F.normalize(features.view(-1, 960), p=2, dim=1)
        cosine_similarities = torch.mm(features_normalized, anchor_normalized.T)

        most_similar_index = torch.argmax(cosine_similarities).item()

        return most_similar_index


class Tracking(Node):
    def __init__(self):
        super().__init__('edi_tracking')

        self.bridge = CvBridge()
        self.twist = Twist()
        
        self.img_sub = self.create_subscription(CompressedImage, "/camera/image_raw/compressed", self.callback, 10)
        self.mode_sub = self.create_subscription(String, "/follow", self.follow_callback, 10)
        self.capture_sub = self.create_subscription(String, '/capturing', self.capture_callback, 10)
        self.img_publisher = self.create_publisher(CompressedImage, '/yolo_video', qos_profile_sensor_data)
        self.crop_publisher = self.create_publisher(CompressedImage, '/crop_img', 10)
        # self.img_sub = self.create_subscription(CompressedImage, "cctv_video", self.callback, qos_profile_sensor_data)
        self.twist_pub = self.create_publisher(Twist, "/base_controller/cmd_vel_unstamped", 10)

        self.one_box_annotator = sv.BoxAnnotator(color = sv.Color.green(), thickness=1, text_thickness=1, text_scale=0.5)
        self.other_box_annotator = sv.BoxAnnotator(color = sv.Color.red(),thickness=1, text_thickness=1, text_scale=0.5)

        self.bridge = CvBridge()

        self.model = YOLO("/home/haneol/pinkbot/src/follow_dl_pkg/follow_dl_pkg/best.pt")

        self.extractor = MobileNetV3FeatureExtractor()
        self.dataset_path = "/home/haneol/pinkbot/install/follow_dl_pkg/share/follow_dl_pkg/img_saved/train"
        # self.dataset_path = os.path.join(get_package_share_directory('follow_dl_pkg'), 'img_saved', 'train', 'one_person')  # 지정한 사람 이미지 샘플 경로

        # self.extractor.calculate_average_feature_vector(self.dataset_path)
        self.capture_finished = False

        self.out_image = None
        self.crosshair_length = 20  # You can adjust this for a smaller or larger crosshair
        
        self.object_center = 0
        self.pre_center = 0

        # self.i_error_a = 0
        # self.d_error_a = 0
        # self.pre_center = 0

        self.follow_mode = False
        self.crop_img = None

        self.capture_finished = False
        self.cropped_images = []


    def delet_person_sample(self, folder_path = '/home/haneol/pinkbot/install/follow_dl_pkg/share/follow_dl_pkg/img_saved/train/one_person/'):

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    pass  # 하위 폴더는 무시 (필요한 경우 추가 처리 가능)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))


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
    
    def capture_callback(self, msg):
        if msg.data == 'capture_stop':
            self.extractor.calculate_average_feature_vector(self.dataset_path)
            self.capture_finished = True
            self.delet_person_sample()

    
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
                if self.capture_finished == True:
                    cropped_images = []
                    for i, bbox in enumerate(detections.xyxy):
                        x_start, x_end = int(bbox[0]), int(bbox[2])
                        y_start, y_end = int(bbox[1]), int(bbox[3])
                        cropped = cv_image[y_start:y_end, x_start:x_end]
                        cropped_images.append(cropped)

                    most_similar_index = self.extractor.find_most_similar(cropped_images)
                    print("가장 유사한 데이터의 인덱스:", most_similar_index)
                    print("탐지된 사람의 수:", detections.xyxy.shape[0])
                    cropped = cropped_images[most_similar_index]


                    one_list = [i == most_similar_index for i in range(len(detections))]
                    other_list = [not i == most_similar_index for i in range(len(detections))]
                    detections_one = detections[np.array(one_list)]
                    detections_other = detections[np.array(other_list)]

                    labels_one = [f"{self.model.model.names[class_id]} {confidence:0.2f}" for _, confidence, class_id, _ in detections_one]
                    labels_other = [f"{self.model.model.names[class_id]} {confidence:0.2f}" for _, confidence, class_id, _ in detections_other]

                    img = self.one_box_annotator.annotate(scene=cv_image, detections=detections_one, labels=labels_one)
                    img = self.other_box_annotator.annotate(scene=img, detections=detections_other, labels=labels_other)
                
                else:
                    cropped = self.get_cropped_img(cv_image, detections.xyxy[0])
                    labels = [f"{self.model.model.names[class_id]} {confidence:0.2f}" for _, confidence, class_id, _ in detections]
                    self.out_image = self.one_box_annotator.annotate(scene=cv_image, detections=detections, labels=labels)
                                
                # cropped = self.get_cropped_img(cv_image, detections.xyxy[most_similar_index])
                self.crop_img = self.bridge.cv2_to_compressed_imgmsg(cropped)
                
                if self.follow_mode == True:
                    x_norm, z_norm = self.normalise_keypoint(cv_image, detections)
                    print("detected area:", z_norm)                
                    
                    self.object_center = x_norm
                    if self.object_center > 0.2:
                        angular = -0.3
                    elif self.object_center < -0.2:
                        angular = 0.3
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