import os
import numpy as np
import math as mt

# 获取根目录下所有的子文件夹
root_dir = '/data3/csi-diffusion/Data/aoyou21ameetingroom/raw-npy'
subfolders = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

def phase_process(phase):
    """
    Process the phase matrix of shape (4, T, 256).
    
    :param phase: A numpy array of shape (4, T, 256) containing phase information.
    :return: A processed phase matrix.
    """
    processed_phase = np.zeros_like(phase)
    processed_phase[0] = phase[0]
    processed_phase[2] = phase[2]
    indices_to_remove = [0, 1, 2, 3, 4, 5, 25, 53, 89, 117, 127, 128, 129, 139, 167, 203, 231, 251, 252, 253, 254, 255]
    indices_to_keep = [i for i in range(phase.shape[2]) if i not in indices_to_remove]
    phase = phase[:, :, indices_to_keep]
    num_antennas, num_timesteps, num_subcarriers = phase.shape
    phase = phase.transpose(0, 2, 1)

    for antenna in [1, 3]:
        phase_data = phase[antenna, :, :]
        phase_unwrapped = np.unwrap(phase_data, axis=0)

        # 相位误差校正
        for tidx in range(1, num_timesteps):
            phase_diff = phase_unwrapped[:, tidx] - phase_unwrapped[:, tidx - 1]
            diff_phase_err = np.diff(phase_diff)
            idxs_invert_up = np.argwhere(diff_phase_err > 0.9 * mt.pi)[:, 0]
            idxs_invert_down = np.argwhere(diff_phase_err < -0.9 * mt.pi)[:, 0]

            for idx_act in idxs_invert_up:
                phase_unwrapped[idx_act + 1:, tidx] -= 2 * mt.pi

            for idx_act in idxs_invert_down:
                phase_unwrapped[idx_act + 1:, tidx] += 2 * mt.pi

        # 第二段代码的逻辑（手动线性校正）
        subcarrier_indices = np.arange(num_subcarriers)
        for tidx in range(num_timesteps):
            k = (phase_unwrapped[-1, tidx] - phase_unwrapped[0, tidx]) / (num_subcarriers - 1)
            b = np.mean(phase_unwrapped[:, tidx]) - k * np.mean(subcarrier_indices)
            phase_unwrapped[:, tidx] -= (k * subcarrier_indices + b)

        processed_phase[antenna, :, indices_to_keep] = phase_unwrapped

    return processed_phase

# 对每个子文件夹进行处理
for subfolder in subfolders:
    # 加载CSI数据和时间戳数据
    amplitude = np.load(os.path.join(subfolder, f'{os.path.basename(subfolder)}_amplitude.npy'))
    # phasediff = np.load(os.path.join(subfolder, f'{os.path.basename(subfolder)}_phase_diff.npy'))
    phase = np.load(os.path.join(subfolder, f'{os.path.basename(subfolder)}_phase.npy'))
    timestamps_sec = np.load(os.path.join(subfolder, f'{os.path.basename(subfolder)}_timestamps_sec.npy'))
    timestamps_usec = np.load(os.path.join(subfolder, f'{os.path.basename(subfolder)}_timestamps_usec.npy'))

    # 将秒和微秒的时间戳组合起来
    timestamps = timestamps_sec + timestamps_usec * 1e-6

    # 获取amplitude元素的形状
    csi_shape = amplitude[0].shape

    # # 创建用于保存数据的文件夹
    # output_folder = os.path.join('/data3/csi-diffusion/Data/aoyou21ameetingroom/single-amp', os.path.basename(subfolder))
    # os.makedirs(output_folder, exist_ok=True)

    # 获取起始时间
    start_timestamp = timestamps[0]

    # 对CSI数据进行分组
    group_id = 0
    group_data = []
    last_group_data = [np.zeros(csi_shape)] * 4  # 初始化最后一组数据为零数组
    expected_group_count = 0
    saved_data = np.zeros((round((timestamps[-1] - start_timestamp) / 0.01) + 1, 4, 256), dtype=np.float32) #round((timestamps[-1] - start_timestamp) / 0.01) + 1

    for i in range(len(amplitude)):
        # 计算应该创建的分组数量
        elapsed_time = timestamps[i] - start_timestamp
        expected_group_count = round(elapsed_time / 0.01)

        # 如果应该创建的分组数量大于当前分组ID
        while expected_group_count > group_id:
            # np.save(os.path.join(output_folder, f'group_{group_id}.npy'), np.stack(last_group_data))
            saved_data[group_id] = last_group_data
            group_data = []
            # 复制 last_group_data 作为新的组
            # np.save(os.path.join(output_folder, f'group_{group_id}.npy'), np.stack(last_group_data))
            group_id += 1

        if expected_group_count < group_id:
            group_data = []
            continue
        
        if len(group_data) == 0 or len(group_data) == 2:
            group_data.append(amplitude[i])
        else:
            group_data.append(phase[i - 1])

        # 如果已经收集了4个数据，就保存当前组的数据
        if len(group_data) == 4:
            # np.save(os.path.join(output_folder, f'group_{group_id}.npy'), np.stack(group_data))
            saved_data[group_id] = group_data
            last_group_data = group_data.copy()  # 更新最后一组数据为当前组数据
            group_id += 1
            group_data = []

    saved_data = np.array(saved_data)
    processed_data = phase_process(saved_data.transpose(1, 0, 2))

    # 创建用于保存打包数据的文件夹
    packed_folder = os.path.join('/data3/csi-diffusion/Data/aoyou21ameetingroom/packed-amp-pha-500-test', os.path.basename(subfolder))
    os.makedirs(packed_folder, exist_ok=True)

    group_data = np.zeros((500, 4, 256), dtype=np.float32)  # 初始化一个大小为256的group，用0填充
    for i in range(processed_data.shape[1]):
        group_data[i % 500] = processed_data[:, i, :]
        
        if (i + 1) % 500 == 0:
            np.save(os.path.join(packed_folder, f'group_{i // 500}.npy'), group_data)
            group_data = np.zeros((500, 4, 256), dtype=np.float32)