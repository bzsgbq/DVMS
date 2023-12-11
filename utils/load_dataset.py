import os
import numpy as np
from torch.utils.data import Dataset
from utils.sampled_dataset import get_video_ids, get_user_ids, get_users_per_video, partition_in_train_val_test_without_video_intersection, partition_in_train_val_test, split_list_by_percentage, partition_in_train_and_test_without_any_intersection, read_sampled_positions_for_trace, read_sampled_playtimes_for_trace
from utils.utils import load_saliency, load_true_saliency, load_fov, transform_batches_cartesian_to_normalized_eulerian, load_dict_from_csv, read_video_pyav, sample_frame_indices, find_closest_elements
from PIL import Image
import av
# from transformers import VideoMAEImageProcessor


MODELS_USING_SALIENCY = ['TRACK', 'Xu_CVPR', 'Nguyen_MM', 'TRACK_salxyz', 'Xu_CVPR_salxyz', 'Nguyen_MM_salxyz', 'perceiver6_salmap', 
                         'TRACK_plus', 'TRACK_res', 'TRACK_convlstm', 'TRACK_posal',
                         'TRACK_ablat_sal', 'TRACK_ablat_fuse', 'TRACK_ablat_all', 'TRACK_left', 'TRACK_sage', 'TRACK_mask', 'TRACK_deform_conv',
                         'pos_only_fuse', 'pos_only_fuse2', 'convlstm', 'gpt4', 'sage', 'informer_plus', 
                         'perceiver', 'perceiver2', 'perceiver3', 'perceiver4', 'perceiver5', 'perceiver6', 'perceiver7', 'perceiver8', 'perceiver9', 'perceiver10',
                         'error_perceiver6']
MODELS_USING_PROCESSED_SALIENCY = ['TRACK_posal', 'perceiver5', 'perceiver6', 'perceiver7', 'perceiver8', 'perceiver9', 'perceiver10', 
                                   'TRACK_salxyz', 'Xu_CVPR_salxyz', 'Nguyen_MM_salxyz', 
                                   'error_perceiver6']
MODELS_USING_MOTION = ['perceiver6_motmap', 'perceiver6_motxyz']
MODELS_USING_PROCESSED_MOTION = ['perceiver6_motxyz']
SLIDE_STRIDE = 5


class CustomDataset(Dataset):

    def __init__(self, model_name, all_traces, all_saliencies, all_motions, all_playtimes, list_IDs, m_window, h_window):
        self.model_name = model_name
        self.all_traces = all_traces
        self.all_saliencies = all_saliencies
        self.all_motions = all_motions
        self.all_playtimes = all_playtimes
        self.list_IDs = list_IDs
        self.m_window = m_window
        self.h_window = h_window
        # self.image_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        user = ID['user']
        video = ID['video']
        x_i = ID['time-stamp']

        # Load the data
        '''方式1: '''
        encoder_pos_inputs = self.all_traces[video][user][x_i-self.m_window : x_i].astype(np.float32)
        decoder_pos_inputs = self.all_traces[video][user][x_i-1 : x_i].astype(np.float32)
        decoder_outputs = self.all_traces[video][user][x_i : x_i+self.h_window].astype(np.float32)

        # print('encoder_pos_inputs.type: ', type(encoder_pos_inputs))  # <class 'numpy.ndarray'>
        # print('encoder_pos_inputs.shape: ', encoder_pos_inputs.shape)  # (5, 3)
        # print('decoder_outputs.type: ', type(decoder_outputs))  # <class 'numpy.ndarray'>
        # print('decoder_outputs.shape: ', decoder_outputs.shape)  # (25, 3)
        # exit(0)

        # plot_points_on_unit_sphere(encoder_pos_inputs, decoder_outputs)
        # exit(0)

        if self.model_name in MODELS_USING_SALIENCY:
            if self.model_name in ['TRACK', 'TRACK_plus', 'TRACK_res', 'TRACK_convlstm', 'TRACK_posal', 
                                   'TRACK_ablat_sal', 'TRACK_ablat_fuse', 'TRACK_ablat_all', 
                                   'TRACK_left', 'TRACK_sage', 'TRACK_mask', 'TRACK_deform_conv', 'convlstm', 
                                   'Xu_CVPR', 'Nguyen_MM', 'Xu_CVPR_salxyz', 'Nguyen_MM_salxyz', 'TRACK_salxyz']:
                m_sal_inputs = np.expand_dims(self.all_saliencies[video][x_i-self.m_window : x_i], axis=-3).astype(np.float32)  # (5, 1, sal_h, sal_w)
                h_sal_inputs = np.expand_dims(self.all_saliencies[video][x_i : x_i+self.h_window], axis=-3).astype(np.float32)
                inputs = [encoder_pos_inputs, m_sal_inputs, decoder_pos_inputs, h_sal_inputs]
                outputs = decoder_outputs
            elif self.model_name in ['pos_only_fuse', 'pos_only_fuse2']:
                h_sal_inputs = np.expand_dims(self.all_saliencies[video][x_i : x_i+self.h_window], axis=-3).astype(np.float32)
                inputs = [encoder_pos_inputs, decoder_pos_inputs, h_sal_inputs]
                outputs = decoder_outputs
            elif self.model_name in ['gpt4', 'sage', 'informer_plus', 'perceiver']:
                h_sal_inputs = np.expand_dims(self.all_saliencies[video][x_i : x_i+self.h_window], axis=-3).astype(np.float32)
                inputs = [encoder_pos_inputs, h_sal_inputs]
                outputs = decoder_outputs
            elif self.model_name in ['perceiver2', 'perceiver3', 'perceiver4']:
                m_sal_inputs = np.expand_dims(self.all_saliencies[video][x_i-self.m_window : x_i], axis=-3).astype(np.float32)  # (5, 1, sal_h, sal_w)
                h_sal_inputs = np.expand_dims(self.all_saliencies[video][x_i : x_i+self.h_window], axis=-3).astype(np.float32)
                inputs = [encoder_pos_inputs, m_sal_inputs, h_sal_inputs]
                outputs = decoder_outputs
            elif self.model_name in ['perceiver5', 'perceiver6', 'perceiver7', 'perceiver8', 'perceiver9', 'perceiver10', 'perceiver6_salmap', 'error_perceiver6']:
                m_sal_inputs = self.all_saliencies[video][x_i-self.m_window : x_i].astype(np.float32)  # (m_window, sal_w, 4)
                h_sal_inputs = self.all_saliencies[video][x_i : x_i+self.h_window].astype(np.float32)
                inputs = [encoder_pos_inputs, m_sal_inputs, h_sal_inputs]
                outputs = decoder_outputs
            elif self.model_name in ['TRACK_posal']:
                m_sal_inputs = self.all_saliencies[video][x_i-self.m_window : x_i].astype(np.float32)  # (m_window, sal_w, 4)
                h_sal_inputs = self.all_saliencies[video][x_i : x_i+self.h_window].astype(np.float32)
                inputs = [encoder_pos_inputs, m_sal_inputs, decoder_pos_inputs, h_sal_inputs]
                outputs = decoder_outputs
            else:
                raise ValueError('Model name not recognized')
        elif self.model_name in MODELS_USING_MOTION:
            if self.model_name in ['perceiver6_motmap', 'perceiver6_motxyz']:
                m_mot_inputs = self.all_motions[video][x_i-self.m_window : x_i].astype(np.float32)
                h_mot_inputs = self.all_motions[video][x_i : x_i+self.h_window].astype(np.float32)
                inputs = [encoder_pos_inputs, m_mot_inputs, h_mot_inputs]
                outputs = decoder_outputs
        else:
            if self.model_name in ['pos_only', 'pos_only_plus', 'pos_only_single', 'Reactive']:
                inputs = [encoder_pos_inputs, decoder_pos_inputs]
                outputs = decoder_outputs
            # elif self.model_name == 'pos_only_origin':
            #     inputs = [transform_batches_cartesian_to_normalized_eulerian(encoder_pos_inputs), transform_batches_cartesian_to_normalized_eulerian(decoder_pos_inputs)]
            #     outputs = transform_batches_cartesian_to_normalized_eulerian(decoder_outputs)
            elif self.model_name in ['LR']:
                inputs = encoder_pos_inputs
                outputs = decoder_outputs
            elif self.model_name == 'mts_mixer':
                inputs = encoder_pos_inputs
                outputs = decoder_outputs
            elif self.model_name == 'informer':
                inputs = encoder_pos_inputs
                outputs = decoder_outputs
            # elif self.model_name == 'VideoMAE':
            #     # region 废弃的两种方式;
            #     # # 方式1: 从视频中读取帧图片;
            #     # end_ms = int(self.all_playtimes[video][user][x_i] * 1000)
            #     # frame_rate = round(float(self.all_videos[video].streams.video[0].average_rate))
            #     # indices = sample_frame_indices(clip_len=16, frame_rate=frame_rate, end_ms=end_ms)
            #     # print(read_video_pyav(self.all_videos[video], indices).shape)
            #     # exit()
            #     # # frames = self.image_processor(list(read_video_pyav(self.all_videos[video], indices)), return_tensors="pt")["pixel_values"]
            #     # print(frames.shape)

            #     # # 方式2: 从文件中读取视频帧图片;
            #     # frames_dir = f'../../datasets/vr-dataset/vid-frames/{video}'
            #     # frames_time_lst = [int(fn.split('.')[0]) for fn in os.listdir(frames_dir)]
            #     # frames_time_lst = np.asarray(frames_time_lst, dtype=int)
            #     # frames_time_lst.sort()
            #     # play_time_lst = self.all_playtimes[video][user][x_i-self.m_window : x_i]
            #     # play_time_lst = np.asarray(play_time_lst*1000, dtype=int)
            #     # play_time_lst = find_closest_elements(play_time_lst, frames_time_lst)
            #     # frames = []
            #     # for pt in play_time_lst:
            #     #     fn = f'{str(pt).zfill(6)}.jpg'
            #     #     frame_path = os.path.join(frames_dir, fn)
            #     #     fra = np.array(Image.open(frame_path))
            #     #     frames.append(fra)
            #     # frames = np.array(frames)

            #     # endregion

            #     # # 方式3: 从all_frames中读取视频帧图片;
            #     # frames = self.all_frames[video][x_i-self.m_window : x_i].astype(np.float32)  # (16, 512, 1024, 3)
                
            #     # # 前3种方式最终的inputs和outputs:
            #     # inputs = [encoder_pos_inputs, frames]
            #     # outputs = decoder_outputs

            #     # 方式4: 从all_fovs中直接读取
            #     fov = self.all_fovs[video][x_i-self.m_window : x_i]  # (16, 224, 224, 3)
            #     fov = self.image_processor(list(fov), return_tensors="pt")["pixel_values"][0]
            #     inputs = [encoder_pos_inputs, fov]
            #     outputs = decoder_outputs
    
            else:
                raise ValueError('Model name not recognized')
        return inputs, outputs, video, user, x_i



class DatasetLoader:
    def __init__(self, dataset_name, model_name, m_window, h_window, init_window=None, end_window=None, provided_videos=True, use_true_saliency=False, tile_h=None, tile_w=None, use_cross_tile_boundary_loss=False):

        self.dataset_name = dataset_name
        self.model_name = model_name
        self.m_window = m_window
        self.h_window = h_window
        self.init_window = init_window
        self.end_window = end_window
        self.provided_videos = provided_videos
        self.use_true_saliency = use_true_saliency
        self.tile_h = tile_h
        self.tile_w = tile_w

        self.root_dataset_folder = os.path.join('./', dataset_name)
        self.video_folder = '../../datasets/vr-dataset/vid-prep'
        self.frame_folder = '../../datasets/vr-dataset/vid-frames'
        self.fov_folder = '../../datasets/vr-dataset/vid-fov'

        self.exp_name = f'_init_{init_window}_in_{m_window}_out_{h_window}_end_{end_window}'
        self.exp_name += f'_{tile_h}x{tile_w}' if tile_h is not None and tile_w is not None else ''
        self.exp_name += f'_useCtbLoss' if use_cross_tile_boundary_loss else ''
        self.sampled_dataset_folder = os.path.join(self.root_dataset_folder, 'sampled_dataset')
        self.true_saliency_folder = os.path.join(self.root_dataset_folder, 'saliency_true')
        self.processed_saliency_folder = os.path.join(self.root_dataset_folder, 'saliency_processed')
        self.saliency_folder = os.path.join(self.root_dataset_folder, 'saliency_paver')
        self.motion_folder = os.path.join(self.root_dataset_folder, 'motion_slof')
        self.processed_motion_folder = os.path.join(self.root_dataset_folder, 'motion_processed')
        self.results_folder = os.path.join(self.root_dataset_folder, self.model_name, 'Results' + self.exp_name)
        self.models_folder = os.path.join(self.root_dataset_folder, self.model_name, 'Models' + self.exp_name)

        self.perc_videos_train = 0.8
        self.perc_users_train = 0.5

    def load(self):
        videos = get_video_ids(self.sampled_dataset_folder)
        users = get_user_ids(self.sampled_dataset_folder)
        users_per_video = get_users_per_video(self.sampled_dataset_folder)

        if self.provided_videos:  # 默认走该分支;
            # if self.dataset_name == 'Xu_CVPR_18':
            #     from Xu_CVPR_18.Read_Dataset import get_videos_train_and_test_from_file
            #     videos_train, videos_test = get_videos_train_and_test_from_file(self.root_dataset_folder)
            #     partition = partition_in_train_val_test_without_video_intersection(self.sampled_dataset_folder, self.init_window, self.end_window, SLIDE_STRIDE, videos_train, videos_test, videos_test, users_per_video)
            # elif self.dataset_name == 'Xu_PAMI_18':
            #     # From Xu_PAMI_18 paper:
            #     # For evaluating the performance of offline-DHP, we randomly divided all 76 panoramic sequences of our PVS-HM database into a training set (61 sequences) and a test set (15 sequences).
            #     # For evaluating the performance of online-DHP [...]. Since the DRL network of offline-DHP was learned over 61 training sequences and used as the initial model of online-DHP, our comparison was conducted on all 15 test sequences of our PVS-HM database.
            #     videos_test = ['KingKong', 'SpaceWar2', 'StarryPolar', 'Dancing', 'Guitar', 'BTSRun', 'InsideCar', 'RioOlympics', 'SpaceWar', 'CMLauncher2', 'Waterfall', 'Sunset', 'BlueWorld', 'Symphony', 'WaitingForLove']
            #     videos_train = ['A380', 'AcerEngine', 'AcerPredator', 'AirShow', 'BFG', 'Bicycle', 'Camping', 'CandyCarnival', 'Castle', 'Catwalks', 'CMLauncher', 'CS', 'DanceInTurn', 'DrivingInAlps', 'Egypt', 'F5Fighter', 'Flight', 'GalaxyOnFire', 'Graffiti', 'GTA', 'HondaF1', 'IRobot', 'KasabianLive', 'Lion', 'LoopUniverse', 'Manhattan', 'MC', 'MercedesBenz', 'Motorbike', 'Murder', 'NotBeAloneTonight', 'Orion', 'Parachuting', 'Parasailing', 'Pearl', 'Predator', 'ProjectSoul', 'Rally', 'RingMan', 'Roma', 'Shark', 'Skiing', 'Snowfield', 'SnowRopeway', 'Square', 'StarWars', 'StarWars2', 'Stratosphere', 'StreetFighter', 'Supercar', 'SuperMario64', 'Surfing', 'SurfingArctic', 'TalkingInCar', 'Terminator', 'TheInvisible', 'Village', 'VRBasketball', 'Waterskiing', 'WesternSichuan', 'Yacht']
            #     partition = partition_in_train_and_test_without_video_intersection(self.sampled_dataset_folder, self.init_window, self.end_window, videos_train, videos_test, users_per_video)
            if self.dataset_name in ['David_MMSys_18', 'Wu_MMSys_17', 'Xu_CVPR_18']:
                train_traces = load_dict_from_csv(os.path.join(self.root_dataset_folder, 'train_set'))
                val_traces = load_dict_from_csv(os.path.join(self.root_dataset_folder, 'val_set'))
                test_traces = load_dict_from_csv(os.path.join(self.root_dataset_folder, 'test_set'))
                partition = partition_in_train_val_test(self.sampled_dataset_folder, self.init_window, self.end_window, self.m_window, self.h_window, SLIDE_STRIDE, train_traces, val_traces, test_traces)
            else:
                raise ValueError(f"Dataset {self.dataset_name} not supported!")
        else:
            videos_train, videos_test = split_list_by_percentage(videos, self.perc_videos_train)
            users_train, users_test = split_list_by_percentage(users, self.perc_users_train)
            partition = partition_in_train_and_test_without_any_intersection(self.sampled_dataset_folder, self.init_window, self.end_window, videos_train, videos_test, users_train, users_test)
        
        all_traces = {}
        for video in videos:
            all_traces[video] = {}
            for user in users_per_video[video]:
                all_traces[video][user] = read_sampled_positions_for_trace(self.sampled_dataset_folder, str(video), str(user))
        
        # Load the saliency only if it's not the position_only baseline
        all_saliencies = {}
        load_saliency = load_true_saliency  # 加载所有种类saliency信息的方式都和加载true_saliency一样;
        if self.model_name in MODELS_USING_PROCESSED_SALIENCY:
            saliency_folder = self.processed_saliency_folder
        elif self.use_true_saliency:
            saliency_folder = self.true_saliency_folder
        else:
            saliency_folder = self.saliency_folder
        for video in videos:
            all_saliencies[video] = load_saliency(saliency_folder, video)
        self.salmap_shape = list(all_saliencies.values())[0].shape[-2:]
        
        all_motions = {}
        # load_motion = load_true_saliency  # 加载motion map的方式和加载true_saliency一样;
        # if self.model_name in MODELS_USING_PROCESSED_MOTION:
        #     for video in videos:
        #         all_motions[video] = load_motion(self.processed_motion_folder, video)
        #     self.motmap_shape = list(all_motions.values())[0].shape[-2:]
        # else:
        #     for video in videos:
        #         all_motions[video] = load_motion(self.motion_folder, video)
        #     self.motmap_shape = list(all_motions.values())[0].shape[-3:-1]
        

        all_playtimes = {}
        for video in videos:
            all_playtimes[video] = {}
            for user in users_per_video[video]:
                all_playtimes[video][user] = read_sampled_playtimes_for_trace(self.sampled_dataset_folder, str(video), str(user))

        # all_frames = {}
        # for video in videos:
        #     all_frames[video] = np.load(os.path.join(self.frame_folder, video + '.npy'))

        # all_fovs = {}
        # for video in videos:
        #     all_fovs[video] = {}
        #     for user in users_per_video[video]:
        #         all_fovs[video][user] = load_fov(self.fov_folder, video, user)

        return  CustomDataset(self.model_name, all_traces, all_saliencies, all_motions, all_playtimes, \
                            #   all_fovs, \
                              partition['train'], self.m_window, self.h_window), \
                CustomDataset(self.model_name, all_traces, all_saliencies, all_motions, all_playtimes, 
                            #   all_fovs, \
                              partition['val'], self.m_window, self.h_window), \
                CustomDataset(self.model_name, all_traces, all_saliencies, all_motions, all_playtimes, 
                            #   all_fovs, \
                              partition['test'], self.m_window, self.h_window)
