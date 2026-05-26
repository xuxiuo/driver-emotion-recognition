# 搭建CNN网络模型  使用kreas
from keras.models import Sequential
#from keras.layers.advanced_activations import PReLU
from keras.layers import PReLU
from keras.layers import Convolution2D, BatchNormalization, AveragePooling2D, Dropout, Flatten, Dense, Activation

def simpleCNN(input_shape, num_classes):
    # 创建一个Sequential模型
    model = Sequential()#用于构建顺序模型，即一层一层地堆叠网络层。

    # 添加一个卷积层，卷积核大小为7x7，步长为1，padding为same，输入形状为input_shape
    model.add(Convolution2D(16, 7, 7, padding='same', input_shape=input_shape))
#Convolution2D：二维卷积层，用于提取图像特征
    # 添加PReLU激活函数
    model.add(PReLU())
    # 添加批归一化层
    model.add(BatchNormalization())
#批归一化层，用于加速训练并提高模型的稳定性。
    # 添加平均池化层，池化大小为5x5，步长为2，padding为same
    model.add(AveragePooling2D(pool_size=(5, 5), strides=(2, 2),padding='same'))
#二维平均池化层，用于下采样特征图
    # 添加Dropout层，dropout率为0.5
    model.add(Dropout(.5))
#Dropout： Dropout层，用于防止过拟合
    # https://blog.csdn.net/qq_39276337/article/details/120451941
    # 添加一个卷积层，卷积核大小为5x5，步长为1，padding为same，输入形状为input_shape
    model.add(Convolution2D(32, 5, 5, padding='same', input_shape=input_shape))
    # 添加PReLU激活函数
    model.add(PReLU())
    # 添加批归一化层
    model.add(BatchNormalization())
    # 添加平均池化层，池化大小为3x3，步长为2，padding为same
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    # 添加Dropout层，dropout率为0.5
    model.add(Dropout(.5))

    # 添加一个卷积层，卷积核大小为3x3，步长为1，padding为same，输入形状为input_shape
    model.add(Convolution2D(32, 3, 3, padding='same', input_shape=input_shape))
    # 添加PReLU激活函数
    model.add(PReLU())
    # 添加批归一化层
    model.add(BatchNormalization())
    # 添加平均池化层，池化大小为3x3，步长为2，padding为same
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    # 添加Dropout层，dropout率为0.5
    model.add(Dropout(.5))

    ## 全连接层
    # 添加Flatten层，将输入展平为一维向量
    model.add(Flatten())
    # 添加一个全连接层，神经元个数为1028
    model.add(Dense(1028))
    # 添加PReLU激活函数
    model.add(PReLU())
    # 添加Dropout层，dropout率为0.5
    model.add(Dropout(.5))
    # 添加PReLU激活函数
    model.add(PReLU())
    # 添加Dropout层，dropout率为0.5
    model.add(Dropout(.5))
    # 添加一个全连接层，神经元个数为num_classes
    model.add(Dense(num_classes))
    # 添加softmax激活函数
    model.add(Activation('softmax'))


    return model



if __name__ == '__main__':
    input_shape = (48, 48, 1)
    #输入为固定尺寸 (48, 48, 1) 的灰度图像（如人脸表情图像）
    model = simpleCNN(input_shape, 7)
#input_shape：输入图像的形状，例如 (48, 48, 1) 表示 48x48 像素的灰度图像。num_classes：输出类别的数量，例如 7 表示有 7 类情绪。
    #输出为 7 类情绪的概率分布（例如：愤怒、厌恶、恐惧、高兴、悲伤、惊讶、中性等）
    model.summary()
    #显示模型结构及参数数量
