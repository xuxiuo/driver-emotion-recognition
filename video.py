import cv2
from keras.models import load_model
from utils.utils import *
from statistics import mode
# 创建一个VideoCapture对象, 会调取摄像头
cap = cv2.VideoCapture(0)


detection_model_path = 'model/haarcascade_frontalface_default.xml'
emotion_model_path = 'model/simple_cnn.hdf5'
#这是一个 字典结构（dict），用于将模型输出的数字编号映射为人类可读的情绪名称
emotions = {
    0: 'anger',  # 生气
    1: 'disgust',  # 厌恶
    2: 'fear',  # 恐惧
    3: 'happy',  # 开心
    4: 'sad',  # 伤心
    5: 'surprised',  # 惊讶
    6: 'normal',  # 中性
}

emotions_window = []
# 加载人脸模型
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
while True:
    # 逐帧捕获
    ret, frame = cap.read()
    # ret 为布尔值, 代表有没有读取到图片, frame 为截取到的一帧率=的图片

    #处理灰度图像相比彩色图像计算量更小，有助于提升实时视频流的处理速度
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 检测人脸
    faces = detect_faces(face_detection, gray)
    for (x, y, w, h) in faces:
        cv2.rectangle(gray, (x,y), (x+w, y+h), (255, 0, 0), 2)
        face = gray[y:y+h, x:x+w]
        try:
            face = cv2.resize(face, (48, 48))
        except:
            print("转换失败")
            continue
        face = preproces_input(face)
        face = np.expand_dims(face, 0)
        face = np.expand_dims(face, -1)
        emotion_label_arg = np.argmax(emotion_classifier.predict(face))

        emotion_text = emotions[emotion_label_arg]

        emotions_window.append(emotion_text)
        # 保持情绪窗口大小不超过10
        if len(emotions_window) >= 10:
            emotions_window.pop(0)
        try:
            emotions_mode = mode(emotions_window) # 选出出现次数最多的
        except:
            continue
        print("emotions_mode", emotions_mode)
        cv2.putText(gray, emotions_mode, (x, y -30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0),
                    2, cv2.LINE_AA)
    # 显示图像
    try:
        cv2.imshow("my_face_emotion_detect", gray)
    except:
        continue
    # 按q键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#结束循环释放VideoCapture对象
cap.release()
cv2.destroyAllWindows()
