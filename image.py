# 图片测试表情识别模型
# 在图片中得到人脸
# 将人脸数据变成 48, 48 传给训练的模型
import cv2
from keras.models import load_model
from utils.utils import *
import numpy as np

test_image_path = 'imgs/img.png'
detection_model_path = 'model/haarcascade_frontalface_default.xml'
#这是一个 预训练的 Haar Cascade 分类器模型文件，用于检测图像中的人脸区域
emotion_model_path = 'model/simple_cnn.hdf5'
emotions = {
    0: 'anger',  # 生气
    1: 'disgust',  # 厌恶
    2: 'fear',  # 恐惧
    3: 'happy',  # 开心
    4: 'sad',  # 伤心
    5: 'surprised',  # 惊讶
    6: 'normal',  # 中性
}

# 加载人脸模型
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

emotion_target_size = emotion_classifier.input_shape[1:3]

# 加载原始图像
rgb_image = load_image(test_image_path, grayscale=False)
gray_image = load_image(test_image_path, grayscale=True)
# 去掉维度为1的维度
gray_image = np.squeeze(gray_image)
gray_image = gray_image.astype('uint8')
print(gray_image.shape)
print(gray_image)

## 检测图片中所有的人脸
faces = detect_faces(face_detection, gray_image)
for face_coordinates in faces:
    x1, x2, y1, y2 = get_coordinates(face_coordinates)
    # 抠出人脸
    gray_face = gray_image[y1:y2, x1:x2]

    try:
        gray_face = cv2.resize(gray_face, (emotion_target_size))
    except:
        print("转换失败")
        continue
    # 归一化
    gray_face = preproces_input(gray_face)
    gray_face = np.expand_dims(gray_face,0)
    # (1, 48, 48, 1)
    gray_face = np.expand_dims(gray_face,-1)
    res = emotion_classifier.predict(gray_face)
    emotion_label_arg = np.argmax(res)
    print("emotion_label_arg", emotion_label_arg)
    emotion_text = emotions[emotion_label_arg]
    print("emotion_text",emotion_text)

    color = ( 0, 0, 255)
    draw_bounding_box(face_coordinates, rgb_image, color)
    draw_text(face_coordinates, rgb_image, emotion_text, color, 0, face_coordinates[3] +30, 1, 2)
bgr_image = cv2.cvtColor(rgb_image,cv2.COLOR_RGB2BGR)
cv2.imwrite('imgs/img2.png', bgr_image)




