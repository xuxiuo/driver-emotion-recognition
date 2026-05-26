# 训练表情识别数据集
from utils.utils import load_data, preproces_input
from model import simpleCNN
from keras.callbacks import ModelCheckpoint, CSVLogger
import os
data_path = 'datasets/fer2013.csv'
model_save_path = "model/simple_cnn.hdf5"#模型保存路径

# 加载数据，函数从 data_path 加载数据，返回两个数组：faces 和 emotions
faces, emotions = load_data(data_path)

# 归一化
faces = preproces_input(faces)
num_classes = emotions.shape[1]#表情标签的数量。
image_size = faces.shape[1:]#输入图像的尺寸


model = simpleCNN(image_size, num_classes)

# 断点续训
if os.path.exists(model_save_path):
    model.load_weights(model_save_path)
    # 加载成功前面保存的参数信息
    print("check_point_loaded")

## 编译模型，使用 adam 优化器和 categorical_crossentropy 损失函数编译模型，并设置评估指标为准确率
model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])

## 记录日志
#将训练过程中的日志记录到 training.log 文件中
csv_logger = CSVLogger('training.log')

# 保存检查点
#ModelCheckpoint在验证集准确率提高时保存模型参数到 model_save_path
model_checkpoint = ModelCheckpoint(model_save_path, 'val_accuracy', verbose=1, save_best_only=True)

model_callbacks = [model_checkpoint, csv_logger]

# 训练;
bacth_size = 128#批次大小
epochs = 1000#训练轮数
model.fit(faces, emotions,bacth_size,
          epochs,verbose=1, callbacks=model_callbacks,
          validation_split=0.1, shuffle=True
          )
#model.fit：开始训练模型，使用 faces 和 emotions 作为输入和标签，设置 batch_size、epochs、verbose、callbacks、validation_split 和 shuffle 参数
