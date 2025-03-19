import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import cv2
import numpy as np
from scipy.ndimage import filters
import tensorflow as tf
from object_detection.utils import config_util
from matplotlib import pyplot as plt
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
import os
from os import path
import roslib   #roslib.load_manifest(PKG)
import rospy
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Accel
from sensor_msgs.msg import CompressedImage
import time
import uuid
import cv2
import mediapipe as mp
import numpy as np
import tf_slim as slim

WORKSPACE_PATH = 'Tensorflow/workspace'
MODEL_PATH = WORKSPACE_PATH + '/models'
CONFIG_PATH = MODEL_PATH + '/my_ssd_mobnet/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH + '/my_ssd_mobnet/'
ANNOTATION_PATH = WORKSPACE_PATH + '/annotations'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH + '/pre-trained-models'

CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
CONFIG_PATH = MODEL_PATH + '/' + CUSTOM_MODEL_NAME + '/pipeline.config'
config = config_util.get_configs_from_pipeline_file(CONFIG_PATH)

pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(CONFIG_PATH, "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)

pipeline_config.model.ssd.num_classes = 7
pipeline_config.train_config.batch_size = 4
pipeline_config.train_config.fine_tune_checkpoint = PRETRAINED_MODEL_PATH + '/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint/ckpt-0'
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path = ANNOTATION_PATH + '/label_map.pbtxt'
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [ANNOTATION_PATH + '/train.record']
pipeline_config.eval_input_reader[0].label_map_path = ANNOTATION_PATH + '/label_map.pbtxt'
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [ANNOTATION_PATH + '/test.record']

config_text = text_format.MessageToString(pipeline_config)
with tf.io.gfile.GFile(CONFIG_PATH, "wb") as f:
    f.write(config_text)

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-6')).expect_partial()


# @tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections
# def callback(data):
#     np_arr = np.fromstring(data.data, np.uint8)
#     image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#     cv2.imwrite('testst.jpg', image_np)
#     cv2.imshow("TurtleView", image_np)
#     print("gettingimage")
def move():
    rospy.init_node('gesture_robot',anonymous=True)
    velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    accel_publisher = rospy.Publisher('/cmd_vel', Accel, queue_size=1)

    # camera_subscriber = rospy.Subscriber('/raspicam_node/image/compressed', CompressedImage, callback)

    vel_msg = Twist()
    vel_msg.linear.x = 0.0
    vel_msg.linear.y = 0
    vel_msg.linear.z = 0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0
    vel_msg.angular.z = 0


    acl_msg = Accel()
    acl_msg.linear.x = 0.0
    acl_msg.linear.y = 0
    acl_msg.linear.z = 0
    acl_msg.angular.x = 0
    acl_msg.angular.y = 0
    acl_msg.angular.z = 0

    velocity_publisher.publish(vel_msg)
    accel_publisher.publish(acl_msg)

    category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH + '/label_map.pbtxt')

    # Setup capture
    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(min_detection_confidence=0.8, max_num_hands=1)
    mpDraw = mp.solutions.drawing_utils
    scale_percent = 80
    previous_time = 0
    current_time = 0
    while True:

        ret, img = cap.read()
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        # img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        frame = np.zeros(shape=img.shape, dtype=np.uint8)
        min_x, min_y = frame.shape[1], frame.shape[0]
        max_x, max_y = 0, 0
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
                mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)
            crop_frame = frame
            image_np = np.array(crop_frame)
            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
            detections = detect_fn(input_tensor)

            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                          for key, value in detections.items()}
            detections['num_detections'] = num_detections

            # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            label_id_offset = 1
            image_np_with_detections = image_np.copy()

            viz_utils.visualize_boxes_and_labels_on_image_array(
                img,
                detections['detection_boxes'],
                detections['detection_classes'] + label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=1,
                min_score_thresh=0.9,
                agnostic_mode=False)

            index = detections['detection_classes'][0] + label_id_offset
            print(index)
            print(category_index.get(index))
        else:
            index = 0;
            print(index)
            print ('autopilot')
        if index == 0 :        # NO SIGNAL stop
            # vel_msg.linear.x = 0
            # vel_msg.linear.y = 0
            # vel_msg.linear.z = 0
            vel_msg.angular.z = 0
            # acl_msg.linear.x = 0.0
            velocity_publisher.publish(vel_msg)
            # accel_publisher.publish(acl_msg)

        if index == 4:          # stop
            vel_msg.linear.x = 0
            vel_msg.linear.y = 0
            vel_msg.linear.z = 0
            vel_msg.angular.x = 0
            vel_msg.angular.y = 0
            vel_msg.angular.z = 0
            velocity_publisher.publish(vel_msg)

        elif index == 1:        # forward
            vel_msg.linear.x = 0.1
            vel_msg.angular.z = 0
            velocity_publisher.publish(vel_msg)
        elif index == 2:    #left
            vel_msg.angular.z = 0.3
            velocity_publisher.publish(vel_msg)
        elif index == 3:    #right
            vel_msg.angular.z = -0.3
            velocity_publisher.publish(vel_msg)
        elif index == 5:    #back
            vel_msg.linear.x = -0.1
            vel_msg.angular.z = 0
            velocity_publisher.publish(vel_msg)
        elif index == 6:    #fast
            if vel_msg.linear.x < 0:
                vel_msg.linear.x -= 0.05
            elif vel_msg.linear.x > 0:
                vel_msg.linear.x += 0.05
            accel_publisher.publish(vel_msg)
        elif index == 7:    #slow
            if vel_msg.linear.x < 0:
                vel_msg.linear.x = vel_msg.linear.x + 0.05
            elif vel_msg.linear.x > 0:
                vel_msg.linear.x = vel_msg.linear.x - 0.05
            accel_publisher.publish(vel_msg)

        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_ITALIC, 2, (255, 0, 0), 3)
        cv2.imshow('object detection', img)
        # cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            break

        # detections = detect_fn(input_tensor)


if __name__ == '__main__':
    try:
        move()
    except rospy.ROSInterruptException: pass
