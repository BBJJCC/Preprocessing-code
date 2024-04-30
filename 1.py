# 从pcap文件中提取csi数据和时间戳，并保存在npy中
import numpy as np
from nexcsi import decoder
import os
import glob
import shutil
import matplotlib.pyplot as plt
import math as mt

device = "rtac86u" # nexus5, nexus6p, rtac86u

pcap_folder = '/data3/csi-diffusion/Data/aoyou21ameetingroom/raw-data'
output_folder = '/data3/csi-diffusion/Data/aoyou21ameetingroom/raw-npy'

pcap_files = glob.glob(os.path.join(pcap_folder, '*.pcap'))

def phase_process(phase):
    """
    Process the phase matrix of shape (4, T, 256).
    
    :param phase: A numpy array of shape (4, T, 256) containing phase information.
    :return: A processed phase matrix.
    """
    processed_phase = np.zeros_like(phase)
    indices_to_remove = [0, 1, 2, 3, 4, 5, 25, 53, 89, 117, 127, 128, 129, 139, 167, 203, 231, 251, 252, 253, 254, 255]
    indices_to_keep = [i for i in range(phase.shape[2]) if i not in indices_to_remove]
    phase = phase[:, :, indices_to_keep]
    num_antennas, num_timesteps, num_subcarriers = phase.shape
    phase = phase.transpose(0, 2, 1)

    for antenna in range(num_antennas):
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

for pcap_filename in pcap_files:
    pcap_name_parts = os.path.splitext(os.path.basename(pcap_filename))[0].split('-')
    save_filename = f"{pcap_name_parts[-2]}-{pcap_name_parts[-1]}"

    samples = decoder(device).read_pcap(pcap_filename)

    timestamps_sec = samples['ts_sec']
    timestamps_usec = samples['ts_usec']
    csi_data = samples['csi']

    csi_data = decoder(device).unpack(samples['csi'], zero_nulls=True, zero_pilots=True)
    amplitude = np.abs(csi_data)
    phase = np.angle(csi_data)
    # if phase.shape[0] % 4:
    #     break
    # phase = phase.reshape(-1, 4, 256)
    # phase = phase.transpose(1, 0, 2)

    # phase = phase_process(phase)

    # phase = phase.transpose(1, 0, 2)

    # T = phase.shape[0] * phase.shape[1]  # 计算 T 的值
    # phase = phase.reshape(T, 256)
    # phase_diff = phase[1::2] - phase[::2]

    # plt.plot(amplitude[0])
    # plt.show()

    save_folder = os.path.join(output_folder, save_filename)
    os.makedirs(save_folder, exist_ok=True)

    np.save(os.path.join(save_folder, f'{save_filename}_timestamps_sec.npy'), timestamps_sec)
    np.save(os.path.join(save_folder, f'{save_filename}_timestamps_usec.npy'), timestamps_usec)
    np.save(os.path.join(save_folder, f'{save_filename}_amplitude.npy'), amplitude)
    # np.save(os.path.join(save_folder, f'{save_filename}_phase_diff.npy'), phase_diff)
    np.save(os.path.join(save_folder, f'{save_filename}_phase.npy'), phase)
