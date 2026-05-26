import cv2#OpenCV，用于图像处理（如保存图像）
import pandas as pd
import numpy as np
import os#文件系统操作（如创建目录）
#将数字标签（0~6）映射为对应的中文/英文情绪名称
emotions = {
    0: 'anger',  # 生气
    1: 'disgust',  # 厌恶
    2: 'fear',  # 恐惧
    3: 'happy',  # 开心
    4: 'sad',  # 伤心
    5: 'surprised',  # 惊讶
    6: 'normal',  # 中性
}
#如果指定路径不存在，则自动创建该路径
def createDir(dir):
    if os.path.exists(dir) == False:
        os.makedirs(dir)
        createDir("../datasets/imgs")

def saveImageFromfer2013(file):
#定义一个函数 saveImageFromfer2013，接收一个 CSV 文件路径作为参数
    faces_data = pd.read_csv(file)#读取 CSV 文件
    print(faces_data.head())
    print(faces_data.columns)
    print(set(faces_data['Usage']))

    for index in range(len(faces_data)):#遍历每一条记录，提取图像信
    # for index in range(5):
        emotion_data = faces_data.iloc[index, 0]

        image_data = faces_data.iloc[index, 1]
        print(image_data)
        print(type(image_data))
        usage_data = faces_data.iloc[index, 2]

        data = list(map(float, image_data.split()))
        # print(len(data))
        image_data = np.array(data).reshape(48, -1)

        dirName = usage_data

        emotion_name = emotions[emotion_data]
        image_path = os.path.join(dirName, emotion_name)
        createDir(image_path)
        dirName = os.path.join(image_path, f'{index}.jpg')
        cv2.imwrite(dirName, image_data)


saveImageFromfer2013("../datasets/fer2013.csv")
print("当前工作目录：", os.getcwd())