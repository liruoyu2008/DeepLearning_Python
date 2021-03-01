from enum import Enum, unique
import os
import shutil
from keras import layers, models, optimizers
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import matplotlib.pyplot as plt


@unique
class AnimalType(Enum):
    """猫狗类型枚举

    Args:
        Enum (str): 传入的枚举值
    """
    Cat = 1
    Dog = 2


def copy_files(src_dir: str, dst_dir: str, start: int, end: int, animal_type: AnimalType):
    """从源文件夹将指定索引开始到指定索引结束的图像拷贝到目标文件夹，动物类型由枚举值给定

    Args:
        src_dir (str): 源文件夹路径
        dst_dir (str): 目标文件夹路径
        start (int): 起始索引（包含）
        end (int): 终止索引（不包含）
        animal_type (AnimalType): 动物类型枚举
    """

    # 确定动物名称
    ani = ''
    if animal_type == AnimalType.Cat:
        ani = 'cat'
    else:
        ani = 'dog'

    # 创建目录，拷贝文件
    if os.path.isdir(dst_dir) is False:
        os.makedirs(dst_dir)
    fnames = [ani+'.{}.jpg'.format(i) for i in range(start, end)]
    for fname in fnames:
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dst_dir, fname)
        shutil.copyfile(src, dst)


def make_small_dataset():
    """创建各数据子集目录（训练、验证、测试）
    """
    copy_files(original_dataset_dir, train_cats_dir, 0,
               1000, AnimalType.Cat)
    copy_files(original_dataset_dir, train_dogs_dir, 0,
               1000, AnimalType.Dog)
    copy_files(original_dataset_dir, validation_cats_dir, 1000,
               1500,  AnimalType.Cat)
    copy_files(original_dataset_dir, validation_dogs_dir, 1000,
               1500, AnimalType.Dog)
    copy_files(original_dataset_dir, test_cats_dir, 1500,
               2000,  AnimalType.Cat)
    copy_files(original_dataset_dir, test_dogs_dir, 1500,
               2000, AnimalType.Dog)


# kaggle上猫🐱，狗🐕数据集解压后的目录（包含训练12500张、测试12500张和csv文件）
original_dataset_dir = r'C:\Users\Ryu\Downloads\dogs-vs-cats'

# 原始数据集的子集基目录路径
base_dir = r'C:\Users\Ryu\Downloads\small'

#　训练集目录
train_dir = os.path.join(base_dir, 'train')
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

#　验证集目录
validation_dir = os.path.join(base_dir, 'validation')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

#　测试集目录
test_dir = os.path.join(base_dir, 'test')
test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')

# 创建各数据子集目录（训练、验证、测试）
# make_small_dataset()

# 构建卷积神经网络
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
print(model.summary())

# 编译网络
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

# 图像生成器,加入图像增强(图像随机变换形态)
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# 生成mini-batch迭代器
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

# 开始训练(从生成器中获取数据)
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=50)

# 保存模型
model.save('cats_and_dogs_small_1.h5')

# 获取与展示训练、验证数据
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
