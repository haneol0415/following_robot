import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
import mediapipe as mp
import rclpy
from rclpy.node import Node
from PIL import Image
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge


TIMER_PERIOD = 0.05

class HandImgSubscriber(Node):
    def __init__(self):
        super().__init__(node_name='hand_subscriber')
        self.bridge = CvBridge()

        self.crop_subscriber = self.create_subscription(
            CompressedImage,
            '/crop_img',
            self.hand_callback,
            10)
        
        self.hand_publisher = self.create_publisher(String, '/hand_gesture', 10)
        
        self.hand = None

        self.model = models.mobilenet_v3_small(pretrained=True)

        num_classes = 3  # 모델의 마지막 레이어 변경 (클래스 수에 맞게), ['normal', 'start', 'stop']
        self.model.classifier[-1] = torch.nn.Linear(self.model.classifier[-1].in_features, num_classes)
        
        weights_path = "/home/haneol/pinkbot/src/follow_dl_pkg/follow_dl_pkg/MobileNetV3_state_dict.pt"
        self.model.load_state_dict(torch.load(weights_path)) # # 저장된 모델 가중치 로드
        self.model.eval()
        
        # self.model.load_state_dict(torch.load("/home/haneol/pinkbot/src/follow_dl_pkg/follow_dl_pkg/MobileNetV3_state_dict.pt"))
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model.to(self.device)

        self.data_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        
        
    def hand_callback(self, data):
        image = self.bridge.compressed_imgmsg_to_cv2(data)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        input_image = self.data_transforms(image).unsqueeze(0)

        # 모델에 이미지 입력 전달
        with torch.no_grad():
            outputs = self.model(input_image)

        # 예측 결과 확인
        _, predicted = torch.max(outputs, 1)
        print(f"Predicted class: {predicted.item()}")

        msg = String()

        if predicted == 0:
            self.hand = 'normal'
        elif predicted == 1:
            self.hand = 'start'
        else:
            self.hand = 'stop'

        msg.data = self.hand

        self.hand_publisher.publish(msg)


        








def main(args=None):
    
    rclpy.init(args=args)
    
    hand_subscriber = HandImgSubscriber()
    
    try :
        rclpy.spin(hand_subscriber)
    
    except KeyboardInterrupt :
        hand_subscriber.get_logger().info('Subscribe Stopped')
    
    finally :
        hand_subscriber.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__' :
    main()