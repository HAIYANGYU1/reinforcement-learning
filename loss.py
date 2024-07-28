import pandas as pd
import matplotlib.pyplot as plt

# 读取txt文件
file_path = '/home/meer/桌面/ppo/train_ppo/model2/progress.txt'
data = pd.read_csv(file_path, sep='\t')

# 选取所需的四个参数
parameters = ['LossPi', 'LossV', 'DeltaLossPi', 'DeltaLossV']
df = data[parameters]

# 生成折线图
plt.figure(figsize=(10, 6))

for parameter in parameters:
    plt.plot(df.index, df[parameter], label=parameter)

plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Loss Parameters Over Epochs')
plt.legend()
plt.grid(True)
plt.show()
