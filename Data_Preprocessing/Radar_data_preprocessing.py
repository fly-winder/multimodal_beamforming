

import glob
import os
import numpy as np
import matplotlib.pyplot as plt


def range_angle_map(data, fft_size=256):
    data = np.fft.fft(data, axis=1)  # Range FFT
    data -= np.mean(data, 2, keepdims=True)
    data = np.fft.fft(data, fft_size, axis=0)  # Angle FFT
    # data = np.fft.fftshift(data, axes=0)
    data = np.abs(data).sum(axis=2)  # Sum over velocity
    return data.T
def range_velocity_map(data):
    data = np.fft.fft(data, axis=1)  # Range FFT
    # data -= np.mean(data, 2, keepdims=True)
    data = np.fft.fft(data, 256, axis=2)  # Velocity FFT
    # data = np.fft.fftshift(data, axes=2)
    data = np.abs(data).sum(axis=0)  # Sum over antennas
    # data = np.log(1 + data)
    return data
def minmax(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())
folder= 'Multi_Modal/scenario33'# Adaptation_dataset_multi_modal 0Multi_Modal Multi_Modal_Test
path="../Dataset/"+folder+"/unit1/radar_data/"
path_ang="../Dataset/"+folder+"/unit1/radar_data_ang/"
path_vel="../Dataset/"+folder+"/unit1/radar_data_vel/"

radarfiles=os.listdir(path)
if not os.path.isdir(path_ang):
    os.mkdir(path_ang)

if not os.path.isdir(path_vel):
    os.mkdir(path_vel)
from joblib import Parallel, delayed
def process(filename):
    print(filename)
    data = np.load(path + filename)
    radar_range_ang_data = range_angle_map(data)
    radar_range_vel_data = range_velocity_map(data)
    np.save(path_ang + filename, minmax(radar_range_ang_data))
    np.save(path_vel + filename, minmax(radar_range_vel_data))
Parallel(n_jobs=10)(delayed(process)(filename) for filename in radarfiles if ".npy" in filename)




















# import glob
# import os
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def range_angle_map(data, fft_size=256):
#     # data = np.fft.fft(data, axis=1)  # Range FFT 4*256*250
#     # data -= np.mean(data, 2, keepdims=True)# 4*256*250
#     # # data = np.fft.fft(data, fft_size, axis=0)  # Angle FFT 256*256*250
#     # # data = np.fft.fftshift(data, axes=0) #  256*256*250
#     # data = np.abs(data).sum(axis=0)  # Sum over velocity 256*256
#     # 新方法
#     # data1 = data[0, :, :]  # n_samoples*N_chirps=信号强度256*250
#     data2 = np.fft.fft(data, axis=2)  # Range FFT n_samples*doppler
#     data3 = data * data2 #256
#     data4 = np.resize(data3,(4,256,256))
#     return data4
# # def range_velocity_map(data):
# #     data = np.fft.fft(data, axis=1)  # Range FFT 4*256*250
# #     data -= np.mean(data, 2, keepdims=True)#4*256*250
# #     data = np.fft.fft(data, 256, axis=2)  # Velocity FFT 4*256*256
# #     # data = np.fft.fftshift(data, axes=2) # 4*256*256
# #     data = np.abs(data).sum(axis=0)  # Sum over antennas 256*256
# #     # data = np.log(1 + data)
# #     return data
# def minmax(arr):
#     return (arr - arr.min()) / (arr.max() - arr.min())
# # folder= 'Adaptation_dataset_multi_modal/scenario33'# Adaptation_dataset_multi_modal 0Multi_Modal Multi_Modal_Test
# folder= 'Multi_Modal/scenario34'#
# path="../Dataset/"+folder+"/unit1/radar_data/"
# # path_ang="../Dataset/"+folder+"/unit1/radar_data_ang/"
# # path_vel="../Dataset/"+folder+"/unit1/radar_data_vel/"
# path_vel="../Dataset/"+folder+"/unit1/radar_data_vel/"
#
# radarfiles=os.listdir(path)
# # if not os.path.isdir(path_ang):
# #     os.mkdir(path_ang)
#
# if not os.path.isdir(path_vel):
#     os.mkdir(path_vel)
# from joblib import Parallel, delayed
# def process(filename):
#     print(filename)
#     data = np.load(path + filename)
#     radar_range_ang_data = range_angle_map(data)
#     # radar_range_vel_data = range_velocity_map(data)
#     np.save(path_vel + filename, minmax(radar_range_ang_data))
#     # np.save(path_vel + filename, minmax(radar_range_vel_data))
# Parallel(n_jobs=10)(delayed(process)(filename) for filename in radarfiles if ".npy" in filename)
# # filename = "radar_data_7.npy"
# # print(filename)
# # data = np.load(path + filename) #4*256*250
# # radar_range_ang_data = range_angle_map(data)#256*256
# # radar_range_vel_data = range_velocity_map(data)# 256*256
# # x = np.arange(0,256)
# # y = np.arange(0,250)
# # x,y = np.meshgrid(x,y)
# # fig = plt.figure()
# # ax3 = plt.axes(projection='3d')
# # ax3.plot_surface(x,y,abs(radar_range_ang_data),cmap='gist_rainbow_r')
# # plt.show()
#
#
