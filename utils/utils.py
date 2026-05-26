import pandas as pd
import numpy as np
import cv2
from keras.preprocessing import image
def load_data(data_file):
    # 读取数据文件
    faces_data = pd.read_csv(data_file)
    # 将pixels列转换为列表
    pixels = faces_data["pixels"].to_list()
    # one-hot
    emotions = pd.get_dummies(faces_data["emotion"]).values
    faces = []
    for pixel_seq in pixels:
        face = list(map(int, pixel_seq.split()))
        face = np.array(face).reshape(48, 48)
        faces.append(face)
    faces = np.array(faces)
#将 faces 列表转换为 NumPy 数组，并添加一个维度，使其形状为 (35887, 48, 48, 1)
    # (35887, 48, 48)
    # (35887, 48, 48, 1)
    faces = np.expand_dims(faces, -1)
    return faces, emotions

# 定义一个函数，用于预处理输入数据
def preproces_input(data):
    # 将输入数据转换为numpy数组，数据类型为float32
    x = np.array(data, dtype=np.float32)
    # 将数据除以255.0，进行归一化处理
    return x/255.0

# 定义一个函数，用于加载图像
def load_image(image_path, grayscale=False, target_size=None):
    # 如果grayscale参数为True，则将color_mode设置为"grayscale"，否则设置为"rgb"
    if grayscale:
        color_mode = "grayscale"
    else:
        color_mode = "rgb"

    # 使用PIL库加载图像，设置颜色模式和目标大小
    pill_image = image.load_img(image_path, color_mode=color_mode, target_size=target_size)
    # 将PIL图像转换为NumPy数组
    return image.img_to_array(pill_image)


def detect_faces(detect_model, gray_image_array):
    # 使用detectMultiScale函数检测灰度图像中的所有人脸
    return detect_model.detectMultiScale(gray_image_array, 1.3, 5)#使用 detect_model 模型检测人脸，返回人脸的坐标
#detect_model：用于检测人脸的模型  gray_image_array：灰度图像的 NumPy 数组
def get_coordinates(face_coordinates):
    # 将传入的face_coordinates参数解包，分别赋值给x, y, width, height
    x, y , width, height = face_coordinates
    # 返回一个元组，包含左上角和右下角的坐标
    return (x, x+width, y, y+height)
def draw_bounding_box(face_coordinates, image_array, color):#绘制边界框
    x, y, width, height = face_coordinates
    cv2.rectangle(image_array, (x, y), (x+width, y+height), color, 2)
# 定义一个绘制文本的函数
def draw_text(face_coordinates, image_array, text, color, x_offset=0, y_offset=0, fonst_scale=2, thickness=2):
    # 获取人脸坐标的x和y值
    x, y = face_coordinates[:2]
    # 在图像上绘制文本
    cv2.putText(image_array, text, (x + x_offset, y+y_offset), cv2.FONT_HERSHEY_SIMPLEX, fonst_scale, color, thickness, cv2.LINE_AA)


if __name__ == '__main__':
    faces, emotions = load_data("../datasets/fer2013.csv")
    print(faces.shape)
    print(emotions.shape)
    print(emotions[0])