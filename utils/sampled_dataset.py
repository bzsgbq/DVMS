import os

import numpy as np
import pandas as pd


# returns the whole data organized as follows:
# time-stamp, x, y, z (in 3d coordinates)
def read_sampled_data_for_trace(sampled_dataset_folder, video, user):
    path = os.path.join(sampled_dataset_folder, str(video), str(user))
    data = pd.read_csv(path, header=None)
    return data.values


# returns only the positions from the trace
# ~time-stamp~ and ~playback-time~ are removed from the output, only x, y, z (in 3d coordinates) is returned
def read_sampled_positions_for_trace(sampled_dataset_folder, video, user):
    path = os.path.join(sampled_dataset_folder, str(video), str(user))
    data = pd.read_csv(path, header=None)
    if 'Wu_MMSys_17' in sampled_dataset_folder:
        return data.values[:, 2:]
    else:
        return data.values[:, 1:]


# returns only the PlaybackTimes from the trace
def read_sampled_playtimes_for_trace(sampled_dataset_folder, video, user):
    path = os.path.join(sampled_dataset_folder, str(video), str(user))
    data = pd.read_csv(path, header=None)
    if 'Wu_MMSys_17' in sampled_dataset_folder:
        return data.values[:, 1]
    else:
        return None


# Returns the ids of the videos in the dataset
def get_video_ids(sampled_dataset_folder):
    list_of_videos = [o for o in os.listdir(sampled_dataset_folder) if not o.endswith('.gitkeep')]
    # Sort to avoid randomness of keys(), to guarantee reproducibility
    list_of_videos.sort()
    return list_of_videos


# returns the unique ids of the users in the dataset
def get_user_ids(sampled_dataset_folder):
    videos = get_video_ids(sampled_dataset_folder)
    users = []
    for video in videos:
        for user in [o for o in os.listdir(os.path.join(sampled_dataset_folder, video)) if not o.endswith('.gitkeep')]:
            users.append(user)
    list_of_users = list(set(users))
    # Sort to avoid randomness of keys(), to guarantee reproducibility
    list_of_users.sort()
    return list_of_users


# Returns a dictionary indexed by video, and under each index you can find the users for which the trace has been stored for this video
def get_users_per_video(sampled_dataset_folder):
    videos = get_video_ids(sampled_dataset_folder)
    users_per_video = {}
    for video in videos:
        users_per_video[video] = [user for user in os.listdir(os.path.join(sampled_dataset_folder, video))]
    return users_per_video


# divides a list into two sublists with the first sublist having samples proportional to "percentage"
def split_list_by_percentage(the_list, percentage):
    # Fixing random state for reproducibility
    np.random.seed(19680801)
    # Shuffle to select randomly
    np.random.shuffle(the_list)
    num_samples_first_part = int(len(the_list) * percentage)
    train_part = the_list[:num_samples_first_part]
    test_part = the_list[num_samples_first_part:]
    return train_part, test_part


# returns a dictionary partition with two indices:
# partition['train'] and partition['test']
# partition['train'] will contain randomly perc_videos_train percent of videos and perc_users_train from each video
# partition['test'] the remaining samples
# the sample consists on a structure of {'video':video_id, 'user':user_id, 'time-stamp':time-stamp_id}
# In this case we don't have any intersection between users nor between videos in train and test sets
# init_window and end_window are used to crop the initial and final indices of the sequence
# e.g. if the indices are [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] and init_window = 3, end_window=2
# the resulting indices of the sequence will be: [3, 4, 5, 6, 7, 8]
# ToDo perform a better split taking into account that a video may be watched by different amounts of users
def partition_in_train_and_test_without_any_intersection(sampled_dataset_folder, init_window, end_window, videos_train,
                                                         videos_test, users_train, users_test):
    partition = {}
    partition['train'] = []
    partition['test'] = []
    for video in videos_train:
        for user in users_train:
            # to get the length of the trace
            try:
                trace_length = read_sampled_data_for_trace(sampled_dataset_folder, video, user).shape[0]
                for tstap in range(init_window, trace_length - end_window):
                    ID = {'video': video, 'user': user, 'time-stamp': tstap}
                    partition['train'].append(ID)
            except FileNotFoundError:
                continue
    for video in videos_test:
        for user in users_test:
            # to get the length of the trace
            try:
                trace_length = read_sampled_data_for_trace(sampled_dataset_folder, video, user).shape[0]
                for tstap in range(init_window, trace_length - end_window):
                    ID = {'video': video, 'user': user, 'time-stamp': tstap}
                    partition['test'].append(ID)
            except FileNotFoundError:
                continue
    return partition


# returns a dictionary partition with two indices:
# partition['train'] and partition['test']
# partition['train'] will contain randomly perc_videos_train percent of videos and perc_users_train from each video
# partition['test'] the remaining samples
# the sample consists on a structure of {'video':video_id, 'user':user_id, 'time-stamp':time-stamp_id}
# In this case the partition is performed only by videos
def partition_in_train_and_test_without_video_intersection(sampled_dataset_folder, init_window, end_window,
                                                           videos_train, videos_test, users_per_video):
    partition = {}
    partition['train'] = []
    partition['test'] = []
    for video in videos_train:
        for user in users_per_video[video]:
            # to get the length of the trace
            trace_length = read_sampled_data_for_trace(sampled_dataset_folder, video, user).shape[0]
            for tstap in range(init_window, trace_length - end_window):
                ID = {'video': video, 'user': user, 'time-stamp': tstap}
                partition['train'].append(ID)
    for video in videos_test:
        for user in users_per_video[video]:
            # to get the length of the trace
            trace_length = read_sampled_data_for_trace(sampled_dataset_folder, video, user).shape[0]
            for tstap in range(init_window, trace_length - end_window):
                ID = {'video': video, 'user': user, 'time-stamp': tstap}
                partition['test'].append(ID)
    return partition


def partition_in_train_val_test_without_video_intersection(sampled_dataset_folder, init_window, end_window, slide_stride, 
                                                           videos_train, videos_val, videos_test, users_per_video):
    partition = {}
    partition['train'] = []
    partition['test'] = []
    partition['val'] = []

    def _gen_partition(kind, videos):
        for video in videos:
            for user in users_per_video[video]:
                # to get the length of the trace
                trace_length = read_sampled_data_for_trace(sampled_dataset_folder, video, user).shape[0]
                for tstap in range(init_window, trace_length - end_window, slide_stride):
                    ID = {'video': video, 'user': user, 'time-stamp': tstap}
                    partition[kind].append(ID)
    
    _gen_partition('train', videos_train)
    _gen_partition('val', videos_val)
    _gen_partition('test', videos_test)
    
    return partition


# returns a dictionary partition with two indices:
# partition['train'] and partition['test']
# partition['train'] will contain the traces in train_traces
# partition['test'] the traces in test_traces
# the sample consists on a structure of {'video':video_id, 'user':user_id, 'time-stamp':time-stamp_id}
# In this case the samples in train and test may belong to the same user or (exclusive or) the same video
def partition_in_train_and_test(sampled_dataset_folder, init_window, end_window, train_traces, test_traces):
    partition = {}
    partition['train'] = []
    partition['test'] = []
    for trace in train_traces:
        user = str(trace[0])
        video = trace[1]
        # to get the length of the trace
        trace_length = read_sampled_data_for_trace(sampled_dataset_folder, video, user).shape[0]
        for tstap in range(init_window, trace_length - end_window):
            ID = {'video': video, 'user': user, 'time-stamp': tstap}
            partition['train'].append(ID)
    for trace in test_traces:
        user = str(trace[0])
        video = trace[1]
        # to get the length of the trace
        trace_length = read_sampled_data_for_trace(sampled_dataset_folder, video, user).shape[0]
        for tstap in range(init_window, trace_length - end_window):
            ID = {'video': video, 'user': user, 'time-stamp': tstap}
            partition['test'].append(ID)
    return partition


# returns a dictionary partition with 3 indices: 'train', 'val', 'test';
# partition['train'] will contain the traces in train_traces
# partition['val'] the traces in val_traces
# partition['test'] the traces in test_traces
# the sample consists on a structure of {'video':video_id, 'user':user_id, 'time-stamp':time-stamp_id}
# In this case the samples in train and test may belong to the same user or (exclusive or) the same video
def partition_in_train_val_test(sampled_dataset_folder, init_window, end_window, m_window, h_window, slide_stride, train_traces, val_traces, test_traces):
    partition = {}
    partition['train'] = []
    partition['val'] = []
    partition['test'] = []
    
    def _get_bad_tstap_set(fovxy_data):
        bad_idxs = np.where((fovxy_data[:, 0] < -1) | (fovxy_data[:, 0] > 1) | (fovxy_data[:, 1] < -1) | (fovxy_data[:, 1] > 1))[0]
        bad_tstaps = []
        for bi in bad_idxs:
            bad_tstaps.extend(list(range(bi - m_window, bi + h_window + 1)))
        bad_tstaps = np.array(bad_tstaps)
        bad_tstaps = bad_tstaps[np.where((bad_tstaps >= 0) & (bad_tstaps < fovxy_data.shape[0]))[0]]  # 删除badd_tstaps中的负数和大于fovxy_data.shape[0]的数;
        bad_tstap_set = set(bad_tstaps)
        return bad_tstap_set

    # # test _get_bad_tstap_set()
    # m_window = 3
    # h_window = 2
    # fovxy_data = np.array([[0, -2], [2, 0], [0.5, 0], [2, -2]])
    # bad_tstap_set = _get_bad_tstap_set(fovxy_data)
    # print(bad_tstap_set)
    # exit(0)

    def _gen_partition(kind, traces):
        for trace in traces:
            user = str(trace[0])
            video = trace[1]
            fovxy_data = read_sampled_data_for_trace(sampled_dataset_folder, video, user)[:, -2:]
            trace_length = fovxy_data.shape[0]
            bad_tstap_set = _get_bad_tstap_set(fovxy_data)  # David_MMSys_18数据集中，bad_tstap_set为空集, 也就是不存在在0.2s内视点直接移动出FoV的情况; 但是在Wu_MMSys_17数据集中，bad_tstap_set不为空集;
            for tstap in range(init_window, trace_length - end_window, slide_stride):
                if tstap in bad_tstap_set:
                    continue
                ID = {'video': video, 'user': user, 'time-stamp': tstap}
                partition[kind].append(ID)

    _gen_partition('train', train_traces)
    _gen_partition('val', val_traces)
    _gen_partition('test', test_traces)    
    
    return partition