import sys
sys.path.insert(0, './')

import os
import numpy as np
import pandas as pd
from utils.utils import eulerian_to_cartesian, cartesian_to_eulerian, get_xyz_grid, salmap2posalfeat, motmap2motxyz, from_position_to_tile_probability_cartesian, rotationBetweenVectors, interpolate_quaternions, degrees_to_radian, radian_to_degrees, compute_orthodromic_distance, store_dict_as_csv
from Wu_MMSys_17.Read_Dataset import sample_dataframe

from pyquaternion import Quaternion
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import argparse
from tqdm import tqdm


ROOT_FOLDER = './Nasrabadi_MMSys_19/dataset/'
OUTPUT_FOLDER_ORIGINAL_XYZ = './Nasrabadi_MMSys_19/original_dataset_xyz'
OUTPUT_FOLDER = './Nasrabadi_MMSys_19/sampled_dataset'
OUTPUT_FOLDER_THETAPHI = './Nasrabadi_MMSys_19/sampled_dataset_thetaphi'
OUTPUT_FOLDER_TRUE_SALIENCY = './Nasrabadi_MMSys_19/saliency_true'
OUTPUT_FOLDER_PAVER_SALIENCY = './Nasrabadi_MMSys_19/saliency_paver_big'
OUTPUT_FOLDER_PROCESSED_SALIENCY = './Nasrabadi_MMSys_19/saliency_processed'
OUTPUT_FOLDER_SLOF_MOTION = './Nasrabadi_MMSys_19/motion_slof_big'
OUTPUT_FOLDER_PROCESSED_MOTION = './Nasrabadi_MMSys_19/motion_processed'

SAMPLING_RATE = 0.2

SALMAP_SHAPE = (128, 256)

VIDEOS = [str(i+1) for i in range(30)]

NUM_SAMPLES_PER_USER = 300


def get_orientations_for_trace(filename):
    dataframe = pd.read_csv(filename, engine='python', header=0, sep=',')
    data = dataframe[[' longitude', ' latitude']]
    return data.values

def get_time_stamps_for_trace(filename):
    dataframe = pd.read_csv(filename, engine='python', header=0, sep=',')
    data = dataframe[' start timestamp']
    return data.values

# returns the frame rate of a video using openCV
# ToDo Copied (changed videoname to videoname+'_saliency' and video_path folder) from Xu_CVPR_18/Reading_Dataset (Author: Miguel Romero)
def get_frame_rate(videoname):
    video_mp4 = videoname+'_saliency.mp4'
    video_path = os.path.join(ROOT_FOLDER, 'content/saliency', video_mp4)
    video = cv2.VideoCapture(video_path)
    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver) < 3:
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    else:
        fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    return fps

# Generate a dataset first with keys per user, then a key per video in the user and then for each sample a set of three keys
# 'sec' to store the time-stamp. 'yaw' to store the longitude, and 'pitch' to store the latitude
def get_original_dataset():
    dataset = {}
    for root, directories, files in tqdm(os.walk(os.path.join(ROOT_FOLDER, 'Traces'))):
        for enum_trace, filename in enumerate(files):
            # print('get head orientations from original dataset traces for video', enum_trace, '/', len(files))
            splitted_filename = filename.split('_')
            user = splitted_filename[0]
            video = splitted_filename[-1].replace('.csv', '')
            if user not in dataset.keys():
                dataset[user] = {}
            if video == 'Intro':
                continue
            
            file_path = os.path.join(root, filename)
            df = pd.read_csv(file_path, engine='python', header=None, sep=',')
            df.columns = ['PlaybackTime', 'Qx', 'Qy', 'Qz', 'Qw', 'Vx', 'Vy', 'Vz']
            
            dataset[user][video] = df
    return dataset

# From "dataset/Videos/Readme_Videos.md"
# Latitude and longitude positions are normalized between 0 and 1 (so they should be multiplied according to the
# resolution of the desired equi-rectangular image output dimension).
# Participants started exploring omnidirectional contents either from an implicit longitudinal center
# (0-degrees and center of the equirectangular projection) or from the opposite longitude (180-degrees).
def transform_the_degrees_in_range(yaw, pitch):
    yaw = yaw*2*np.pi
    pitch = pitch*np.pi
    return yaw, pitch

# Performs the opposite transformation than transform_the_degrees_in_range
# Transform the yaw values from range [0, 2pi] to range [0, 1]
# Transform the pitch values from range [0, pi] to range [0, 1]
def transform_the_radians_to_original(yaw, pitch):
    yaw = yaw/(2*np.pi)
    pitch = pitch/np.pi
    return yaw, pitch


# # ToDo Copied exactly from AVTrack360/Reading_Dataset (Author: Miguel Romero)
# def create_sampled_dataset(original_dataset, rate):
#     dataset = {}
#     for enum_user, user in enumerate(original_dataset.keys()):
#         dataset[user] = {}
#         for enum_video, video in enumerate(original_dataset[user].keys()):
#             print('creating sampled dataset', 'user', enum_user, '/', len(original_dataset.keys()), 'video', enum_video, '/', len(original_dataset[user].keys()))
#             sample_orig = np.array([1, 0, 0])
#             data_per_video = []
#             for sample in original_dataset[user][video]:
#                 sample_yaw, sample_pitch = transform_the_degrees_in_range(sample['yaw'], sample['pitch'])
#                 sample_new = eulerian_to_cartesian(sample_yaw, sample_pitch)
#                 quat_rot = rotationBetweenVectors(sample_orig, sample_new)
#                 # append the quaternion to the list
#                 data_per_video.append([sample['sec'], quat_rot[0], quat_rot[1], quat_rot[2], quat_rot[3]])
#                 # update the values of time and sample
#             # interpolate the quaternions to have a rate of 0.2 secs
#             data_per_video = np.array(data_per_video)
#             dataset[user][video] = interpolate_quaternions(data_per_video[:, 0], data_per_video[:, 1:], rate=rate)
#     return dataset


# version2: 直接读取已生成好的original_dataset_xyz, 并且简单上采样 (不进行插值);
def create_sampled_dataset(original_dataset_xyz_path, rate):
    dataset = {}
    for root, directories, files in tqdm(os.walk(original_dataset_xyz_path)):
        if len(directories) == 0:  # 说明当前的root下只有文件, 没有目录;
            for filename in files:
                video = root.split('/')[-1]
                user = filename
                if user not in dataset.keys():
                    dataset[user] = {}
                file_path = os.path.join(root, filename)
                df = pd.read_csv(file_path, engine='python', header=None, names=['pt', 'x', 'y', 'z'], sep=',')

                # 降采样: 方式2: 以pt为标准进行降采样;
                df = sample_dataframe(df, 'pt', rate)
                
                df = df[:NUM_SAMPLES_PER_USER]
                dataset[user][video] = df
    return dataset

# ToDo Copied exactly from AVTrack360/Reading_Dataset (Author: Miguel Romero)
def recover_original_angles_from_quaternions_trace(quaternions_trace):
    angles_per_video = []
    orig_vec = np.array([1, 0, 0])
    for sample in quaternions_trace:
        quat_rot = Quaternion(sample[1:])
        sample_new = quat_rot.rotate(orig_vec)
        restored_yaw, restored_pitch = cartesian_to_eulerian(sample_new[0], sample_new[1], sample_new[2])
        restored_yaw, restored_pitch = transform_the_radians_to_original(restored_yaw, restored_pitch)
        angles_per_video.append(np.array([restored_yaw, restored_pitch]))
    return np.array(angles_per_video)

def recover_original_angles_from_xyz_trace(xyz_trace):
    angles_per_video = []
    for sample in xyz_trace:
        restored_yaw, restored_pitch = cartesian_to_eulerian(sample[1], sample[2], sample[3])
        restored_yaw, restored_pitch = transform_the_radians_to_original(restored_yaw, restored_pitch)
        angles_per_video.append(np.array([restored_yaw, restored_pitch]))
    return np.array(angles_per_video)

# ToDo Copied exactly from Xu_PAMI_18/Reading_Dataset (Author: Miguel Romero)
def recover_xyz_from_quaternions_trace(quaternions_trace):
    angles_per_video = []
    orig_vec = np.array([1, 0, 0])
    for sample in quaternions_trace:
        quat_rot = Quaternion(sample[1:])
        sample_new = quat_rot.rotate(orig_vec)
        angles_per_video.append(sample_new)
    return np.concatenate((quaternions_trace[:, 0:1], np.array(angles_per_video)), axis=1)


# ToDo Copied exactly from AVTrack360/Reading_Dataset (Author: Miguel Romero)
# Return the dataset
# yaw = 0, pitch = pi/2 is equal to (1, 0, 0) in cartesian coordinates
# yaw = pi/2, pitch = pi/2 is equal to (0, 1, 0) in cartesian coordinates
# yaw = pi, pitch = pi/2 is equal to (-1, 0, 0) in cartesian coordinates
# yaw = 3*pi/2, pitch = pi/2 is equal to (0, -1, 0) in cartesian coordinates
# yaw = Any, pitch = 0 is equal to (0, 0, 1) in cartesian coordinates
# yaw = Any, pitch = pi is equal to (0, 0, -1) in cartesian coordinates
def get_xyz_dataset(sampled_dataset):
    dataset = {}
    for user in sampled_dataset.keys():
        dataset[user] = {}
        for video in sampled_dataset[user].keys():
            dataset[user][video] = recover_xyz_from_quaternions_trace(sampled_dataset[user][video])
    return dataset


# Store the dataset into the folder_to_store
def store_dataset(dataset, folder_to_store):
    for user in dataset.keys():
        for video in dataset[user].keys():
            video_folder = os.path.join(folder_to_store, video)
            os.makedirs(video_folder, exist_ok=True)  # Create the folder for the video if it doesn't exist
            path = os.path.join(video_folder, user)
            df = pd.DataFrame(dataset[user][video])
            df.to_csv(path, header=False, index=False)


def compare_integrals(original_dataset, sampled_dataset):
    error_per_trace = []
    traces = []
    for user in original_dataset.keys():
        for video in original_dataset[user].keys():
            integ_yaws_orig = 0
            integ_pitchs_orig = 0
            for count, sample in enumerate(original_dataset[user][video]):
                if count == 0:
                    prev_sample = original_dataset[user][video][0]
                else:
                    dt = sample['sec'] - prev_sample['sec']
                    integ_yaws_orig += sample['yaw'] * dt
                    integ_pitchs_orig += sample['pitch'] * dt
                    prev_sample = sample
            angles_per_video = recover_original_angles_from_quaternions_trace(sampled_dataset[user][video])
            integ_yaws_sampl = 0
            integ_pitchs_sampl = 0
            for count, sample in enumerate(angles_per_video):
                if count == 0:
                    prev_time = sampled_dataset[user][video][count, 0]
                else:
                    dt = sampled_dataset[user][video][count, 0] - prev_time
                    integ_yaws_sampl += angles_per_video[count, 0] * dt
                    integ_pitchs_sampl += angles_per_video[count, 1] * dt
                    prev_time = sampled_dataset[user][video][count, 0]
            error_per_trace.append(np.sqrt(np.power(integ_yaws_orig-integ_yaws_sampl, 2) + np.power(integ_pitchs_orig-integ_pitchs_sampl, 2)))
            traces.append({'user': user, 'video': video})
    return error_per_trace, traces

### Check if the quaternions are good
def compare_sample_vs_original(original_dataset, sampled_dataset):
    for user in original_dataset.keys():
        for video in original_dataset[user].keys():
            pitchs = []
            yaws = []
            times = []
            for sample in original_dataset[user][video]:
                times.append(sample['sec'])
                yaws.append(sample['yaw'])
                pitchs.append(sample['pitch'])
            angles_per_video = recover_original_angles_from_xyz_trace(sampled_dataset[user][video])
            plt.subplot(1, 2, 1)
            plt.plot(times, yaws, label='yaw')
            plt.plot(times, pitchs, label='pitch')
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.plot(sampled_dataset[user][video][:, 0], angles_per_video[:, 0], label='yaw')
            plt.plot(sampled_dataset[user][video][:, 0], angles_per_video[:, 1], label='pitch')
            plt.legend()
            plt.show()

# ToDo Copied exactly from AVTrack360/Reading_Dataset (Author: Miguel Romero)
def plot_3d_trace(positions, user, video):
    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v), alpha=0.1, color="r")
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label='parametric curve')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('User: %s, Video: %s' % (user, video))
    plt.show()

# ToDo Copied exactly from AVTrack360/Reading_Dataset (Author: Miguel Romero)
# def plot_all_traces_in_3d(xyz_dataset, error_per_trace, traces):
#     indices = np.argsort(-np.array(error_per_trace))
#     for trace_id in indices:
#         trace = traces[trace_id]
#         user = trace['user']
#         video = trace['video']
#         plot_3d_trace(xyz_dataset[user][video][:, 1:])

def plot_all_traces_in_3d(xyz_dataset):
    for user in xyz_dataset.keys():
        for video in xyz_dataset[user].keys():
            plot_3d_trace(xyz_dataset[user][video][:, 1:], user, video)


def create_and_store_sampled_dataset(plot_comparison=False, plot_3d_traces=False):
    # # 方式1: 从头生成original_dataset
    # original_dataset = get_original_dataset()
    # sampled_dataset = create_sampled_dataset(original_dataset, rate=SAMPLING_RATE)
    # 方式2: 直接读取, 而不再重新生成:
    sampled_dataset = create_sampled_dataset(OUTPUT_FOLDER_ORIGINAL_XYZ, rate=SAMPLING_RATE)
    if plot_comparison:
        compare_sample_vs_original(original_dataset, sampled_dataset)
    xyz_dataset = sampled_dataset#get_xyz_dataset(sampled_dataset)  # 由于我们没有使用原代码中的插值函数, 所以sampled_dataset中就是三元组而不是四元组;
    if plot_3d_traces:
        plot_all_traces_in_3d(xyz_dataset)
    store_dataset(xyz_dataset, OUTPUT_FOLDER)


def create_and_store_thetaphi_dataset():
    dataset = load_sampled_dataset()
    print("creating and storing thetaphi dataset...")
    thetaphi_dataset = {}
    for user in tqdm(dataset.keys()):
        thetaphi_dataset[user] = {}
        for video in dataset[user].keys():
            # sampled_dataset.columns: ['playback_time(s)', 'x', 'y', 'z']
            # thetaphi_dataset.columns: ['playback_time(s)', 'theta', 'phi' ]
            thetaphi_dataset[user][video] = dataset[user][video][:, :3].copy()
            thetaphi_dataset[user][video][:, 1], thetaphi_dataset[user][video][:, 2] = cartesian_to_eulerian(dataset[user][video][:, 1], dataset[user][video][:, 2], dataset[user][video][:, 3])
            if thetaphi_dataset[user][video].shape[0] < NUM_SAMPLES_PER_USER:
                print(f'Warning: thetaphi_dataset[{user}][{video}].shape[0] < NUM_SAMPLES_PER_USER')
    store_dataset(thetaphi_dataset, OUTPUT_FOLDER_THETAPHI)
    print("thetaphi dataset created and stored!")


def create_and_store_true_saliency(sampled_dataset):
    if not os.path.exists(OUTPUT_FOLDER_TRUE_SALIENCY):
        os.makedirs(OUTPUT_FOLDER_TRUE_SALIENCY)
    xyz_grid = get_xyz_grid(*SALMAP_SHAPE, lib='numpy')
    for enum_video, video in enumerate(VIDEOS):
        print('creating true saliency for video', video, '-', enum_video, '/', len(VIDEOS))
        real_saliency_for_video = []
        for x_i in range(NUM_SAMPLES_PER_USER):
            tileprobs_for_video_cartesian = []
            for user in sampled_dataset.keys():
                tileprobs_cartesian = from_position_to_tile_probability_cartesian(pos=sampled_dataset[user][video][x_i, 1:], xyz_grid=xyz_grid, way=1)
                tileprobs_for_video_cartesian.append(tileprobs_cartesian)
            tileprobs_for_video_cartesian = np.array(tileprobs_for_video_cartesian)
            real_saliency_cartesian = np.sum(tileprobs_for_video_cartesian, axis=0) / tileprobs_for_video_cartesian.shape[0]
            real_saliency_for_video.append(real_saliency_cartesian)
        real_saliency_for_video = np.array(real_saliency_for_video)
        true_sal_out_file = os.path.join(OUTPUT_FOLDER_TRUE_SALIENCY, video)
        np.save(true_sal_out_file, real_saliency_for_video)

def load_sampled_dataset():
    list_of_videos = [o for o in os.listdir(OUTPUT_FOLDER) if not o.endswith('.gitkeep')]
    dataset = {}
    for video in list_of_videos:
        for user in [o for o in os.listdir(os.path.join(OUTPUT_FOLDER, video)) if not o.endswith('.gitkeep')]:
            if user not in dataset.keys():
                dataset[user] = {}
            path = os.path.join(OUTPUT_FOLDER, video, user)
            data = pd.read_csv(path, header=None)
            dataset[user][video] = data.values
    return dataset

def get_most_salient_points_per_video():
    from skimage.feature import peak_local_max
    most_salient_points_per_video = {}
    for video in VIDEOS:
        saliencies_for_video_file = os.path.join(OUTPUT_FOLDER_TRUE_SALIENCY, video+'.npy')
        saliencies_for_video = np.load(saliencies_for_video_file)
        most_salient_points_in_video = []
        for id, sal in enumerate(saliencies_for_video):
            coordinates = peak_local_max(sal, exclude_border=False, num_peaks=5)
            coordinates_normalized = coordinates / np.array(SALMAP_SHAPE)
            coordinates_radians = coordinates_normalized * np.array([np.pi, 2.0*np.pi])
            cartesian_pts = np.array([eulerian_to_cartesian(sample[1], sample[0]) for sample in coordinates_radians])
            most_salient_points_in_video.append(cartesian_pts)
        most_salient_points_per_video[video] = np.array(most_salient_points_in_video)
    return  most_salient_points_per_video

def predict_most_salient_point(most_salient_points, current_point):
    pred_window_predicted_closest_sal_point = []
    for id, most_salient_points_per_fut_frame in enumerate(most_salient_points):
        distances = np.array([compute_orthodromic_distance(current_point, most_sal_pt) for most_sal_pt in most_salient_points_per_fut_frame])
        closest_sal_point = np.argmin(distances)
        predicted_closest_sal_point = most_salient_points_per_fut_frame[closest_sal_point]
        pred_window_predicted_closest_sal_point.append(predicted_closest_sal_point)
    return pred_window_predicted_closest_sal_point

def most_salient_point_baseline(dataset):
    most_salient_points_per_video = get_most_salient_points_per_video()
    error_per_time_step = {}
    for enum_user, user in enumerate(dataset.keys()):
        for enum_video, video in enumerate(dataset[user].keys()):
            print('computing error for user', enum_user, '/', len(dataset.keys()), 'video', enum_video, '/', len(dataset[user].keys()))
            trace = dataset[user][video]
            for x_i in range(5, 75):
                model_prediction = predict_most_salient_point(most_salient_points_per_video[video][x_i+1:x_i+25+1], trace[x_i, 1:])
                for t in range(25):
                    if t not in error_per_time_step.keys():
                        error_per_time_step[t] = []
                    error_per_time_step[t].append(compute_orthodromic_distance(trace[x_i+t+1, 1:], model_prediction[t]))
    for t in range(25):
        print(t*0.2, np.mean(error_per_time_step[t]))

def create_original_dataset_xyz(original_dataset):
    dataset = {}
    for enum_user, user in tqdm(enumerate(original_dataset.keys())):
        dataset[user] = {}
        for enum_video, video in enumerate(original_dataset[user].keys()):
            # print('creating original dataset in format for', 'user', enum_user, '/', len(original_dataset.keys()), 'video', enum_video, '/', len(original_dataset[user].keys()))
            df = original_dataset[user][video]
            df2 = pd.DataFrame(columns=['pt', 'x', 'y', 'z'])
            df2['pt'] = df['PlaybackTime']
            df2['x'] = -df['Vz']
            df2['y'] = -df['Vx']
            df2['z'] = df['Vy']
            dataset[user][video] = df2
    return dataset

def create_and_store_original_dataset():
    original_dataset = get_original_dataset()
    original_dataset_xyz = create_original_dataset_xyz(original_dataset)
    store_dataset(original_dataset_xyz, OUTPUT_FOLDER_ORIGINAL_XYZ)

def get_traces_for_train_and_test():
    # Fixing random state for reproducibility
    np.random.seed(7)

    videos = ['1_PortoRiverside', '2_Diner', '3_PlanEnergyBioLab', '4_Ocean', '5_Waterpark', '6_DroneFlight',
              '7_GazaFishermen', '8_Sofa', '9_MattSwift', '10_Cows', '11_Abbottsford', '12_TeatroRegioTorino',
              '13_Fountain', '14_Warship', '15_Cockpit', '16_Turtle', '17_UnderwaterPark', '18_Bar', '19_Touvet']
    users = np.arange(57)

    # Select at random the users for each set
    np.random.shuffle(users)
    num_train_users = int(len(users) * 0.5)
    users_train = users[:num_train_users]
    users_test = users[num_train_users:]

    videos_ids_train = [1, 3, 5, 7, 8, 9, 11, 14, 16, 18]
    videos_ids_test = [0, 2, 4, 13, 15]

    train_traces = []
    for video_id in videos_ids_train:
        for user_id in users_train:
            train_traces.append({'video': videos[video_id], 'user': str(user_id)})

    test_traces = []
    for video_id in videos_ids_test:
        for user_id in users_test:
            test_traces.append({'video': videos[video_id], 'user': str(user_id)})

    return train_traces, test_traces


def get_traces_for_train_val_test():
    # Fixing random state for reproducibility
    np.random.seed(7)

    videos = ['1_PortoRiverside', '2_Diner', '3_PlanEnergyBioLab', '4_Ocean', '5_Waterpark', '6_DroneFlight',
              '7_GazaFishermen', '8_Sofa', '9_MattSwift', '10_Cows', '11_Abbottsford', '12_TeatroRegioTorino',
              '13_Fountain', '14_Warship', '15_Cockpit', '16_Turtle', '17_UnderwaterPark', '18_Bar', '19_Touvet']
    users = np.arange(57)

    # Select at random the users for each set
    np.random.shuffle(users)
    num_train_users = int(len(users) * 0.5)
    users_train = users[:num_train_users]
    users_test = users[num_train_users:]

    # region 方式3: 只划分用户, 不划分视频; 训练集/验证集/测试集的用户均不重叠;
    # 按照4:1:1的比例划分训练集/验证集/测试集:
    users_train = []
    users_val = []
    users_test = []
    for i in range(len(users)):
        if i % 6 == 2:
            users_val.append(users[i])
        elif i % 6 == 5:
            users_test.append(users[i])
        else: # i % 6 == 0 or 1 or 3 or 4
            users_train.append(users[i])

    train_traces = []
    for video_id in videos:
        for user_id in users_train:
            train_traces.append({'video': video_id, 'user': str(user_id)})
    
    val_traces = []
    for video_id in videos:
        for user_id in users_val:
            val_traces.append({'video': video_id, 'user': str(user_id)})
    
    test_traces = []
    for video_id in videos:
        for user_id in users_test:
            test_traces.append({'video': video_id, 'user': str(user_id)})
    # endregion

    return train_traces, val_traces, test_traces


def create_and_store_processed_saliency():
    if not os.path.exists(OUTPUT_FOLDER_PROCESSED_SALIENCY):
        os.makedirs(OUTPUT_FOLDER_PROCESSED_SALIENCY)
    for enum_video, video in enumerate(VIDEOS):
        print('creating processed saliency for video', video, '-', enum_video, '/', len(VIDEOS))
        true_sal_path = os.path.join(OUTPUT_FOLDER_PAVER_SALIENCY, video+'.npy')
        true_sal = np.load(true_sal_path)  # (NUM_FRAMES, 224, 448)

        # # plot:
        # raw_salmap = true_sal[444]
        # import matplotlib as mpl
        # mpl.use('Agg')
        # from matplotlib import pyplot as plt
        # data = raw_salmap.reshape(-1)  # plot_data是将raw_salmap转化为一维数据
        # # 基于seaborn绘制cdf图:
        # sns.ecdfplot(data=data)
        # plt.savefig('test.png')
        # plt.cla()
        # exit()

        # from PIL import Image
        # img = Image.fromarray((raw_salmap * 255).astype('uint8'), mode='L')
        # img.save('test.jpg')
        # exit()

        # 需要将true_saliency设置为paver_big; 其每张salmap的形状为(224, 448)
        H, W = true_sal.shape[-2:]
        num_data = 20480  # 20480 / (224*448) = 0.20408...; 即取每张salmap中saliency值最大的的前20%个数据
        stride = 40  # 以40的步长取数据; 所以最终每张salmap中只取了20480/40=512个数据
        xyz_grid = get_xyz_grid(H, W).numpy()  # (H, W, 3)
        processed_sal = salmap2posalfeat(xyz_grid, true_sal)  # (NUM_FRAMES, H*W, 4)
        processed_sal = np.array([x[np.argsort(x[:, -1])[-num_data:]] for x in processed_sal])[:, ::-stride, :]  # 对于每一帧, 只在H*W个数据中保留saliency值最大的num_data个数据, 并且以stride的步长取数据; (saliency: processed_sal[:, :, -1])     

        # # # test:
        # H, W = 2, 4
        # num_data = 5
        # stride = 2
        # # 随机生成一个processed_sal, 用于测试:
        # processed_sal = np.random.rand(1, H*W, 4)
        # print(processed_sal)
        # processed_sal = np.array([x[np.argsort(x[:, -1])[-num_data:]] for x in processed_sal])[:, ::-stride, :]
        # print(processed_sal)
        # exit()
        
        processed_sal_path = os.path.join(OUTPUT_FOLDER_PROCESSED_SALIENCY, video+'.npy')
        np.save(processed_sal_path, processed_sal)


def create_and_store_processed_motion():
    if not os.path.exists(OUTPUT_FOLDER_PROCESSED_MOTION):
        os.makedirs(OUTPUT_FOLDER_PROCESSED_MOTION)
    for enum_video, video in enumerate(VIDEOS):
        print('creating processed motion for video', video, '-', enum_video, '/', len(VIDEOS))
        motmaps_path = os.path.join(OUTPUT_FOLDER_SLOF_MOTION, video+'.npy')
        motmaps = np.load(motmaps_path)  # (NUM_FRAMES, 320, 640, 3)
        H, W = motmaps.shape[1], motmaps.shape[2]
        num_data = 40960  # 40960 / (320*640) = 0.2 ; 即取每张motmap中saliency值最大的的前20%个数据
        stride = 80  # 以80的步长取数据; 所以最终每张motmap中只取了40960/80=512个数据
        xyz_grid = get_xyz_grid(H, W).numpy()  # (H, W, 3)
        processed_motion = motmap2motxyz(xyz_grid, motmaps)  # (NUM_FRAMES, H*W, 6)
        # 对于每一帧, 只在H*W个数据中保留最后3个分量的平方和最大的num_data个数据, 并且以stride的步长取数据:
        processed_motion = np.array([x[np.argsort(np.sum(x[:, -3:]**2, axis=-1))[-num_data:]] for x in processed_motion])[:, ::-stride, :]
        
        # # test:
        # H, W = 2, 4
        # num_data = 5
        # stride = 2
        # # 随机生成一个processed_motion, 用于测试:
        # processed_motion = np.random.rand(1, H*W, 6)
        # print(processed_motion)
        # print(np.sum(processed_motion[0][:, -3:]**2, axis=-1))
        # processed_motion = np.array([x[np.argsort(np.sum(x[:, -3:]**2, axis=-1))[-num_data:]] for x in processed_motion])[:, ::-stride, :]
        # print(processed_motion)
        # exit()

        processed_motion_path = os.path.join(OUTPUT_FOLDER_PROCESSED_MOTION, video+'.npy')
        np.save(processed_motion_path, processed_motion)


def split_traces_and_store():
    train_traces, val_traces, test_traces = get_traces_for_train_val_test()
    store_dict_as_csv('Nasrabadi_MMSys_19/train_set', ['user', 'video'], train_traces)
    store_dict_as_csv('Nasrabadi_MMSys_19/val_set', ['user', 'video'], val_traces)
    store_dict_as_csv('Nasrabadi_MMSys_19/test_set', ['user', 'video'], test_traces)


if __name__ == "__main__":
    #print('use this file to create sampled dataset or to create true_saliency or to create original dataset in xyz format')

    parser = argparse.ArgumentParser(description='Process the input parameters to parse the dataset.')
    parser.add_argument('--split_traces', action="store_true", dest='_split_traces_and_store', help='Flag that tells if we want to create the files to split the traces into train, val and test.')
    parser.add_argument('--creat_orig_dat', action="store_true", dest='_create_original_dataset', help='Flag that tells if we want to create and store the original dataset.')
    parser.add_argument('--creat_samp_dat', action="store_true", dest='_create_sampled_dataset', help='Flag that tells if we want to create and store the sampled dataset.')
    parser.add_argument('--creat_thph_dat', action="store_true", dest='_create_thetaphi_dataset', help='Flag that tells if we want to create and store the thetaphi dataset.')
    parser.add_argument('--creat_true_sal', action="store_true", dest='_create_true_saliency', help='Flag that tells if we want to create and store the ground truth saliency.')
    parser.add_argument('--compare_traces', action="store_true", dest='_compare_traces', help='Flag that tells if we want to compare the original traces with the sampled traces.')
    parser.add_argument('--plot_3d_traces', action="store_true", dest='_plot_3d_traces', help='Flag that tells if we want to plot the traces in the unit sphere.')
    parser.add_argument('--creat_proc_sal', action="store_true", dest='_create_processed_saliency', help='Flag that tells if we want to create and store the processed saliency.')
    parser.add_argument('--creat_proc_mot', action="store_true", dest='_create_processed_motion', help='Flag that tells if we want to create and store the processed motion.')
    args = parser.parse_args()

    if args._split_traces_and_store:  # 划分出train set和test set, 其实就是将属于train set的视频编号和用户编号存到一个叫做train_set的文件中, 将test set的视频编号和用户编号存到一个叫做test_set的文件中
        split_traces_and_store()

    if args._create_original_dataset:
        create_and_store_original_dataset()  # 这个函数的作用实际上是生成和存储 original_dataset_xyz

    if args._create_sampled_dataset:
        create_and_store_sampled_dataset()

    if args._create_thetaphi_dataset:
        create_and_store_thetaphi_dataset()
    
    if args._compare_traces:
        original_dataset = get_original_dataset()
        sampled_dataset = load_sampled_dataset()
        compare_sample_vs_original(original_dataset, sampled_dataset)

    if args._plot_3d_traces:
        sampled_dataset = load_sampled_dataset()
        plot_all_traces_in_3d(sampled_dataset)

    if args._create_true_saliency:
        if os.path.isdir(OUTPUT_FOLDER):
            sampled_dataset = load_sampled_dataset()
            create_and_store_true_saliency(sampled_dataset)
        else:
            print('Please verify that the sampled dataset has been created correctly under the folder', OUTPUT_FOLDER)

    if args._create_processed_saliency:
        create_and_store_processed_saliency()

    if args._create_processed_motion:
        create_and_store_processed_motion()

    # sampled_dataset = load_sampled_dataset()
    # most_salient_point_baseline(sampled_dataset)
