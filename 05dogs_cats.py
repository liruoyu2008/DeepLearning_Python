import os
import shutil

# kaggle上猫狗数据集解压后的目录（包含训练12500张、测试12500张和csv文件）
original_dataset_dir=r'C:\Users\Ryu\Downloads\dogs-vs-cats'

# 保存较小数据集的目录
base_dir = r'C:\Users\Ryu\Downloads\dogs-vs-cats\cats_and_dogs_small'
os.mkdir(base_dir)