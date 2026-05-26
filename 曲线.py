import pandas as pd
import matplotlib.pyplot as plt

# 读取训练日志文件
log_path = 'training.log'
log_df = pd.read_csv(log_path)

# 设置绘图风格
plt.style.use('ggplot')

# 创建画布
plt.figure(figsize=(14, 5))

# 子图 1: Loss 曲线
plt.subplot(1, 2, 1)
plt.plot(log_df['epoch'], log_df['loss'], label='Train Loss', marker='o')
plt.plot(log_df['epoch'], log_df['val_loss'], label='Val Loss', marker='o')
plt.title('Training and Validation Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# 子图 2: Accuracy 曲线
plt.subplot(1, 2, 2)
plt.plot(log_df['epoch'], log_df['accuracy'], label='Train Accuracy', marker='o')
plt.plot(log_df['epoch'], log_df['val_accuracy'], label='Val Accuracy', marker='o')
plt.title('Training and Validation Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# 保存图像到本地文件
plt.tight_layout()
plt.savefig('training_curve.png')  # 保存为图片文件
print("图像已保存为 training_curve.png")

# 可选：是否同时显示图像？
# plt.show()
