# 将数据分为训练集和测试集
import os
import random
import shutil

# 设置文件夹路径
folder_path = "/home/bjc/paper3-code/Data/aoyou21ameetingroom/packed-amp"  # 替换为包含所有标签文件夹的父文件夹路径
train_path = "/home/bjc/paper3-code/Data/aoyou21ameetingroom/train-amp"
test_path = "/home/bjc/paper3-code/Data/aoyou21ameetingroom/test-amp"

# 设置训练集和测试集比例
train_ratio = 0.7
test_ratio = 1 - train_ratio

# 获取所有标签文件夹
label_folders = os.listdir(folder_path)

# 遍历每个标签文件夹
for label_folder in label_folders:
    label_folder_path = os.path.join(folder_path, label_folder)

    # 检查是否为文件夹
    if os.path.isdir(label_folder_path):
        # 获取当前标签文件夹中所有.npy文件
        npy_files = [f for f in os.listdir(label_folder_path) if f.endswith(".npy")]

        # 计算训练集和测试集的文件数量
        num_files = len(npy_files)
        num_train = int(num_files * train_ratio)
        num_test = num_files - num_train

        # 打乱文件列表顺序
        random.shuffle(npy_files)

        # 创建新的训练集和测试集文件夹
        train_folder = os.path.join(train_path, label_folder)
        test_folder = os.path.join(test_path, label_folder)
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)

        # 将文件复制到训练集文件夹
        for i in range(num_train):
            src_path = os.path.join(label_folder_path, npy_files[i])
            dst_path = os.path.join(train_folder, npy_files[i])
            shutil.copyfile(src_path, dst_path)

        # 将文件复制到测试集文件夹
        for i in range(num_train, num_files):
            src_path = os.path.join(label_folder_path, npy_files[i])
            dst_path = os.path.join(test_folder, npy_files[i])
            shutil.copyfile(src_path, dst_path)

        print("已处理标签文件夹:", label_folder)

print("文件分割完成")
