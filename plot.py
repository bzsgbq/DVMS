import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns
plt.rcParams.update({
    'font.size': 16,
    'font.sans-serif': 'Times New Roman',
    })
import numpy as np
import pandas as pd
import os
import argparse
import sys
sys.path.append('./')
from tqdm import tqdm
from utils.sampled_dataset import get_video_ids, get_user_ids, get_users_per_video, read_sampled_positions_for_trace
from utils.utils import get_bests_worsts_vuxi


parser = argparse.ArgumentParser(description='Plot the models error.')

parser.add_argument('--dataset_name', action='store', dest='dataset_name', help='The name of the dataset used to train this network.')
parser.add_argument('--init_window', action='store', dest='init_window', help='(Optional) Initial buffer window (to avoid stationary part).', type=int)
parser.add_argument('--m_window', action='store', dest='m_window', help='Past history window.', type=int)
parser.add_argument('--h_window', action='store', dest='h_window', help='Prediction window.', type=int)
parser.add_argument('--end_window', action='store', dest='end_window', help='(Optional) Final buffer (to avoid having samples with less outputs).', type=int)
parser.add_argument('--metric', action="store", dest='metric', help='Which metric to use, by default, f1 score is used.')
# metric 可选: 'orthodromic_distance', 'accuracy', 'precision', 'recall', 'f1_score'

args = parser.parse_args()
RATE = 0.2
dataset_name = args.dataset_name
# Buffer window in timesteps
M_WINDOW = args.m_window
# Forecast window in timesteps (5 timesteps = 1 second) (Used in the network to predict)
H_WINDOW = args.h_window
# Initial buffer (to avoid stationary part)
if args.init_window is None:
    INIT_WINDOW = M_WINDOW
else:
    INIT_WINDOW = args.init_window
# final buffer (to avoid having samples with less outputs)
if args.end_window is None:
    END_WINDOW = H_WINDOW
else:
    END_WINDOW = args.end_window

met = args.metric if args.metric is not None else 'f1_score'

SAMPLED_DATASET_FOLDER = os.path.join('./', dataset_name, 'sampled_dataset')
OUTPUT_FOLDER = f'./{dataset_name}/figures/m{M_WINDOW}-h{H_WINDOW}'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


line_style_dict = {
    'Reactive': '-',
    'LR': '-',
    'pos_only': '-',
    'Xu_CVPR': '-.',
    'Nguyen_MM': ':',
    'TRACK': '--',
    'perceiver6': '-',

    'Xu_CVPR_salxyz': '-.',
    'Nguyen_MM_salxyz': ':',
    'TRACK_salxyz': '--',
    'perceiver6_salmap': '-',

    'perceiver6_motmap': '-',
    'perceiver6_motxyz': '-',

    'perceiver5': '-',

    'error_perceiver6': '-',
}

marker_dict = {
    'Reactive': 'v',
    'LR': '^',
    'pos_only': 's',
    'Xu_CVPR': '',
    'Nguyen_MM': '',
    'TRACK': '',
    'perceiver6': 'o',
    
    'Xu_CVPR_salxyz': 'o',
    'Nguyen_MM_salxyz': 'o',
    'TRACK_salxyz': 'o',
    'perceiver6_salmap': '',

    'perceiver6_motmap': '',
    'perceiver6_motxyz': '',

    'perceiver5': '',

    'error_perceiver6': '',
}
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
color_dict = {
    'Reactive': '#9467bd',
    'LR': '#8c564b',
    'pos_only': '#e377c2',

    'Xu_CVPR': '#2ca02c',
    'Xu_CVPR_salxyz': '#2ca02c',
    'Nguyen_MM': '#ff7f0e',
    'Nguyen_MM_salxyz': '#ff7f0e',
    'TRACK': '#d62728',
    'TRACK_salxyz': '#d62728',

    'perceiver6': '#1f77b4',
    'perceiver6_salmap': '#1f77b4',

    'perceiver6_motmap': '#bcbd22',
    'perceiver6_motxyz': '#17becf',

    'perceiver5': '#7f7f7f',

    'error_perceiver6': '#1f77b0',
}

# '''
# Part 1
# '''
# # region: 绘制orth-dist.pdf

# mn_dict = {
#     'Reactive': 'static-baseline',
#     'LR': 'linear-regression',
#     'pos_only': 'deep-pos-only',
#     'Xu_CVPR': 'Xu_CVPR',
#     'Nguyen_MM': 'Nguyen_MM',
#     'TRACK': 'TRACK',
#     'perceiver6': 'Rainbow-VP (ours)',
    
#     'perceiver6_salmap': 'Rainbow-VP-salmap',
#     'perceiver6_motmap': 'Rainbow-VP-motmap',
#     'perceiver6_motxyz': 'Rainbow-VP-motxyz',
    
#     'perceiver5': 'perceiver5',

#     'error_perceiver6': 'Eror-Rainbow-VP',
# }
# plot_model_names = ['Reactive', 'LR', 'pos_only', 'Nguyen_MM', 'Xu_CVPR', 'TRACK', 'perceiver6']
# # plot_model_names = ['pos_only', 'TRACK', 'perceiver6', 'perceiver6_motmap', 'perceiver6_motxyz']
# # plot_model_names = ['pos_only', 'perceiver6']
# # plot_model_names = ['pos_only', 'Nguyen_MM', 'Xu_CVPR', 'TRACK', 'perceiver6', 'perceiver5']
# # plot_model_names = ['perceiver6', 'error_perceiver6']
# for mn in plot_model_names:
#     df = pd.read_csv(f'./{dataset_name}/{mn}/Results_init_{INIT_WINDOW}_in_{M_WINDOW}_out_{H_WINDOW}_end_{END_WINDOW}/errors.csv')
#     avg_error_per_timestep = []
#     for t in range(H_WINDOW):
#         error = df[df['t']==t][met].mean()
#         avg_error_per_timestep.append(error)
#     # 输出当前model 2s-5s (15 time steps) 的平均error:
#     print(f'{mn_dict[mn]}: {np.mean(avg_error_per_timestep[-15:])}')
#     plt.plot(np.arange(1, H_WINDOW+1)*RATE, avg_error_per_timestep, 
#              label=mn_dict[mn], color=color_dict[mn], 
#              linestyle=line_style_dict[mn], linewidth=2.0, 
#              marker=marker_dict[mn], markevery=3, markersize=5.0)

# print(1-1.016344869551145/1.0607724415240634)
# print('-'*20)


# # plt.title(f'Prediction Error ({dataset_name})')
# plt.ylabel('Avg. Orthodromic Distance')
# plt.xlabel('Prediction step (sec.)')
# # plt.ylim(0, 0.8)
# # 自定义labels的位置:
# plt.legend(loc='lower right', bbox_to_anchor=(1.1, 0.0))
# plt.grid()
# # plt.show()
# plot_path = os.path.join(f'{OUTPUT_FOLDER}/orth-dist.pdf')
# plt.savefig(plot_path, 
#             bbox_inches='tight', 
#             transparent=True,
#             pad_inches=0)
# plt.clf()
# # endregion



# '''
# Part 2
# '''
# # region: 绘制running-time.pdf
# running_time_dict = {
#     'Reactive': 0.03886222839355469,
#     'LR': 6.832361221313477,
#     'pos_only': 396.3948029738206,

#     'Xu_CVPR': 390.2729107783391,
#     'Nguyen_MM': 407.52623631404,
#     'TRACK': 408.52893315828766,
#     'perceiver6': 26.960226205679085,

#     # 'Xu_CVPR_salxyz': 395.1108272259052, 
#     # 'Nguyen_MM_salxyz': 397.7462328397311,
#     # 'TRACK_salxyz': 408.57769892765924, 
#     # 'perceiver6_salmap': 25.990871282724235, 
# }

# # 绘制运行时间的柱形图: (使用running_time_dict的keys作为key去取mn_dict的value作为画图时的keys)
# plt.bar([mn_dict[mn] for mn in running_time_dict.keys()], running_time_dict.values(), width=0.5, color=[color_dict[mn] for mn in running_time_dict.keys()])
# # 显示数据标签
# for a,b in zip([mn_dict[mn] for mn in running_time_dict.keys()], running_time_dict.values()):
#     plt.text(a,b, 
#             f'{b:.2f}',  # 保留两位小数:
#              ha='center', 
#              va='bottom',
#             )
# plt.ylabel('Avg. Running Time (ms)')
# plt.xlabel('Algorithm')
# plt.xticks(rotation=30)
# plt.grid()
# plot_path = os.path.join(f'{OUTPUT_FOLDER}/running-time.pdf')
# plt.savefig(plot_path,
#             bbox_inches='tight',
#             transparent=True,
#             pad_inches=0)
# plt.clf()

# print(running_time_dict['perceiver6']/running_time_dict['pos_only'])
# # endregion



# '''
# Part 3
# '''
# # 微观分析: perceiver具体好在哪;
# # region
# mn1 = 'perceiver6'
# df1 = pd.read_csv(f'./{dataset_name}/{mn1}/Results_init_{INIT_WINDOW}_in_{M_WINDOW}_out_{H_WINDOW}_end_{END_WINDOW}/errors.csv')
# mn2 = 'Nguyen_MM'
# df2 = pd.read_csv(f'./{dataset_name}/{mn2}/Results_init_{INIT_WINDOW}_in_{M_WINDOW}_out_{H_WINDOW}_end_{END_WINDOW}/errors.csv')
# # df.columns: ['video', 'user', 'x_i', 't', 'orthodromic_distance', ...]
# # 按照每个video, 每个user, 每个x_i, 求平均的orthodromic_distance:
# df1 = df1.groupby(['video', 'user', 'x_i']).mean().reset_index()
# df2 = df2.groupby(['video', 'user', 'x_i']).mean().reset_index()
# # 计算df1-df2的差值, 并从小到大排序; (越小说明perceiver6越好)
# df_diff = df1.copy()
# df_diff[met] = df1[met] - df2[met]
# df_diff = df_diff.sort_values(by=[met])
# print(df_diff.columns)
# # 输出df_diff的前15行和后5行的video, user, x_i的值; 以list的形式输出;
# print(df_diff[['video', 'user', 'x_i']].head(15).values.tolist())
# print(df_diff[['video', 'user', 'x_i']].tail(5).values.tolist())

# # endregion



# '''
# Part 4
# '''
# # region: 绘制trajectories:

# def get_trajectories_folder(model_name):
#     return f'./plot_trajectories_figures/{dataset_name}/{model_name}/init_{INIT_WINDOW}_in_{M_WINDOW}_out_{H_WINDOW}_end_{END_WINDOW}'

# def get_trajectories_df(model_name, video, user, x_i):
#     file_path = f'{get_trajectories_folder(model_name)}/video_{video}_user_{user}_x_i_{x_i}.csv'
#     df = pd.read_csv(file_path)
#     return df

# def draw_trajectory(theta, phi, color, label, flag_draw_labels):
#     x = theta.values / np.pi
#     y = 1 - phi.values / np.pi
#     # plt.plot(x, y, color=color, label=label, linewidth=1)#, marker='o', markevery=3, markersize=5.0, markerfacecolor='none')
#     if flag_draw_labels:
#         plt.arrow(x[0], y[0], x[1]-x[0], y[1]-y[0], width=0.003, length_includes_head=True, head_width=0.02, head_length=0.03, color=color, label=label)
#     for i in range(1 if flag_draw_labels else 0, len(x)-1):
#         diff_x = x[i+1]-x[i]
#         if abs(diff_x+2) < abs(diff_x):
#             diff_x += 2
#         elif abs(diff_x-2) < abs(diff_x):
#             diff_x -= 2
#         diff_y = y[i+1]-y[i]
#         plt.arrow(x[i], y[i], diff_x, diff_y, width=0.003, length_includes_head=True, head_width=0.02, head_length=0.03, color=color)
#     # plt.arrow(x[-2], y[-2], x[-1]-x[-2], y[-1]-y[-2], width=0.0001, length_includes_head=True, head_width=0.005, head_length=0.01, color=color)

# # 1. pos_only, 
# # 2. nguyen, 
# # 3. rainbow
# bests = [['7_GazaFishermen', 3, 45], 
#          ['5_Waterpark', 15, 35], 
#          ['5_Waterpark', 6, 30], # 3 > 1 > 2
#          ['4_Ocean', 50, 65], # NOTE: 3 > 1 > 2
#          ['7_GazaFishermen', 3, 50], 
#          ['18_Bar', 3, 65], #  3 > 1,2
#          ['18_Bar', 3, 70], # NOTE: 3 > 1,2
#          ['8_Sofa', 50, 30], # NOTE: 3 > 2 > 1
#          ['9_MattSwift', 48, 30], 
#          ['8_Sofa', 18, 65], # 3 > 1,2
#          ['9_MattSwift', 16, 70], # 3 > 1,2
#          ['12_TeatroRegioTorino', 48, 60], 
#          ['9_MattSwift', 51, 60], 
#          ['14_Warship', 51, 60], # NOTE: 2 > 3 > 1
#          ['6_DroneFlight', 50, 40]]

# worsts= [['9_MattSwift', 50, 50], 
#          ['9_MattSwift', 12, 50], 
#          ['9_MattSwift', 50, 45], 
#          ['18_Bar', 12, 40], 
#          ['2_Diner', 18, 30]]

# flag_draw_labels = False

# for vuxi in bests+worsts:
#     video, user, x_i = vuxi
#     print(f'Generating {video}-{user}-{x_i}...')
#     df_rainbow = get_trajectories_df('perceiver6', video, user, x_i)
#     df_nguyen = get_trajectories_df('Nguyen_MM', video, user, x_i)
#     df_pos_only = get_trajectories_df('pos_only', video, user, x_i)

#     draw_trajectory(df_pos_only['theta'][M_WINDOW+H_WINDOW : M_WINDOW+H_WINDOW+H_WINDOW], df_pos_only['phi'][M_WINDOW+H_WINDOW : M_WINDOW+H_WINDOW+H_WINDOW], 
#                     color_dict['pos_only'], 'prediction of HT-only model', flag_draw_labels)

#     draw_trajectory(df_rainbow['theta'][:M_WINDOW], df_rainbow['phi'][:M_WINDOW], 
#                     'grey', 'historical viewpoint trajectory (HT)', flag_draw_labels)
#     draw_trajectory(df_rainbow['theta'][M_WINDOW:M_WINDOW+H_WINDOW], df_rainbow['phi'][M_WINDOW:M_WINDOW+H_WINDOW], 
#                     'black', 'ground-truth future trajectory', flag_draw_labels)
#     # draw_trajectory(df_rainbow['theta'][:M_WINDOW+H_WINDOW], df_rainbow['phi'][:M_WINDOW+H_WINDOW], 
#     #                 'black', 'ground-truth viewpoint trajectory', flag_draw_labels)
    
#     # draw_trajectory(df_nguyen['theta'][M_WINDOW+H_WINDOW : M_WINDOW+H_WINDOW+H_WINDOW], df_nguyen['phi'][M_WINDOW+H_WINDOW : M_WINDOW+H_WINDOW+H_WINDOW], color_dict['Nguyen_MM'], 'HT+VC1', flag_draw_labels)
#     # draw_trajectory(df_rainbow['theta'][M_WINDOW+H_WINDOW : M_WINDOW+H_WINDOW+H_WINDOW], df_rainbow['phi'][M_WINDOW+H_WINDOW : M_WINDOW+H_WINDOW+H_WINDOW], 
#     #                 color_dict['perceiver6'], 'HT+VC model', flag_draw_labels)
    
#     # 设定x轴和y轴的范围:
#     plt.xlim(0, 2)
#     plt.ylim(0, 1)
#     plt.gca().set_aspect(1)
#     plt.axis('off')
    
#     if flag_draw_labels:
#         plt.legend(ncol=1)
        

#     output_folder = os.path.join(OUTPUT_FOLDER, 'trajectories')
#     os.makedirs(output_folder, exist_ok=True)
#     plot_path = os.path.join(f'{output_folder}/{video}-{user}-{x_i}.pdf')
#     plt.savefig(plot_path,
#                 bbox_inches='tight',
#                 transparent=True,
#                 pad_inches=0)
#     plt.clf()
# # endregion




# '''
# Part 5
# '''
# import cv2
# # region: 得到salmap和frame:
# saliency_folder = os.path.join('./', dataset_name, 'true_saliency_paver_big')
# video_folder = os.path.join('./', dataset_name, 'dataset', 'Videos', 'Stimuli')
# SAMPLING_RATE = 0.2
# NUM_FRAMES = 100
# needs = [['4_Ocean', 50, 65], # NOTE: 3 > 1 > 2
#          ['18_Bar', 3, 70], # NOTE: 3 > 1,2
#          ['8_Sofa', 50, 30], # NOTE: 3 > 2 > 1
#          ['14_Warship', 51, 60]] # NOTE: 2 > 3 > 1
# for vuxi in needs:
#     video, user, x_i = vuxi
#     # get salmap:
#     salmaps_path = os.path.join(saliency_folder, video + '.npy')
#     salmaps = np.load(salmaps_path)
#     sal_out_folder = os.path.join(OUTPUT_FOLDER, 'salmaps', f'{video}-{user}-{x_i}')
#     os.makedirs(sal_out_folder, exist_ok=True)
#     for t in range(-M_WINDOW, H_WINDOW):
#         salmap = salmaps[t+x_i]
        
#         # 创建透明度数组
#         # alpha = np.ones_like(salmap)
#         # alpha = np.sin(salmap)
#         alpha = salmap
#         # alpha[salmap < 0.4] = 0

#         # 正向
#         # cmap_name = 'hot'
#         # cmap_name = 'summer'
#         cmap_name = 'afmhot'

#         # 反向
#         salmap = 1 - salmap
#         cmap_name = 'YlGnBu'

#         cmap = plt.colormaps.get_cmap(cmap_name)
#         salmap = cmap(salmap)

#         # 设置透明度
#         salmap[..., -1] = alpha

#         # 绘制热力图
#         plt.imshow(salmap)

#         plt.axis('off')
#         plt.savefig(os.path.join(sal_out_folder, f'{t}.pdf'),
#                     bbox_inches='tight',
#                     transparent=True,
#                     pad_inches=0)
#         plt.clf()
    
#     # get frame:
#     video_path = os.path.join(video_folder, video + '.mp4')
#     cap = cv2.VideoCapture(video_path)
    
#     fps_file = open(os.path.join(video_folder, 'fps'), 'r')
#     # fps_file中, 每个视频的fps信息占一行, 先是视频名, 然后是fps, 以空格分隔;
#     fps = float(fps_file.readlines()[int(video.split('_')[0])-1].split(' ')[1])
#     fps_file.close()
#     print(f'the fps of {video} is {fps}')

#     frame_out_folder = os.path.join(OUTPUT_FOLDER, 'frames')
#     os.makedirs(frame_out_folder, exist_ok=True)
#     frame_idx = x_i * SAMPLING_RATE * fps
#     cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
#     ret, frame = cap.read()
#     cv2.imwrite(os.path.join(frame_out_folder, f'{video}-{user}-{x_i}.png'), frame)
#     cap.release()
#     cv2.destroyAllWindows()
# # endregion




# '''
# Part 6
# '''
# # region: 绘制 salxyz-orth-dist.pdf 和 salxyz-model-size.pdf:
# mn_dict = {
#     'Xu_CVPR': 'Xu_CVPR(salmap)',
#     'Nguyen_MM': 'Nguyen_MM(salmap)',
#     'TRACK': 'TRACK(salmap)',
#     'perceiver6': 'Rainbow-VP(salxyz)',

#     'Xu_CVPR_salxyz': 'Xu_CVPR(salxyz)',
#     'Nguyen_MM_salxyz': 'Nguyen_MM(salxyz)',
#     'TRACK_salxyz': 'TRACK(salxyz)',
#     'perceiver6_salmap': 'Rainbow-VP(salmap)',
# }

# plot_model_names = [
#     'Nguyen_MM', 'Nguyen_MM_salxyz', 
#     'Xu_CVPR', 'Xu_CVPR_salxyz', 
#     'TRACK', 'TRACK_salxyz', 
#     'perceiver6_salmap', 'perceiver6', 

#     # 'Xu_CVPR_salxyz', 
#     # 'Nguyen_MM_salxyz', 
#     # 'TRACK_salxyz', 
#     # 'perceiver6', 
# ]

# for mn in plot_model_names:
#     df = pd.read_csv(f'./{dataset_name}/{mn}/Results_init_{INIT_WINDOW}_in_{M_WINDOW}_out_{H_WINDOW}_end_{END_WINDOW}/errors.csv')
#     avg_error_per_timestep = []
#     for t in range(H_WINDOW):
#         error = df[df['t']==t][met].mean()
#         avg_error_per_timestep.append(error)
#     # 输出当前model的平均error:
#     print(f'{mn_dict[mn]}: {np.mean(avg_error_per_timestep[-15:])}')
#     plt.plot(np.arange(1, H_WINDOW+1)*RATE, avg_error_per_timestep, 
#              label=mn_dict[mn], color=color_dict[mn],
#              linestyle=line_style_dict[mn], linewidth=2.0, 
#              marker=marker_dict[mn], markevery=3, markersize=5.0)

# print('-'*20)

# # plt.title(f'Prediction Error ({dataset_name})')
# plt.ylabel('Avg. Orthodromic Distance')
# plt.xlabel('Prediction step (sec.)')
# # plt.ylim(0, 0.8)
# # 自定义labels的位置:
# plt.legend(loc='lower right', bbox_to_anchor=(1.1, 0.0))
# plt.grid()
# # plt.show()
# plot_path = os.path.join(f'{OUTPUT_FOLDER}/salxyz-orth-dist.pdf')
# plt.savefig(plot_path, 
#             bbox_inches='tight', 
#             transparent=True,
#             pad_inches=0)
# plt.clf()




# # # 绘制模型大小的柱形图: (使用model_size_dict的keys作为key去取mn_dict的value作为画图时的keys)
# model_size_dict = {
#     'Nguyen_MM': 19182727/1024/1024,  # 19.2 MB (19,182,727 bytes)
#     'Nguyen_MM_salxyz': 12891271/1024/1024,  # 12.9 MB (12,891,271 bytes)
#     'Xu_CVPR': 15534215/1024/1024, 
#     'Xu_CVPR_salxyz': 9242759/1024/1024,  # 9.2 MB (9,242,759 bytes)
#     'TRACK': 90574343/1024/1024,  # 90.6 MB (90,574,343 bytes)
#     'TRACK_salxyz': 40242695/1024/1024,  # 40.2 MB (40,242,695 bytes)
#     # 'perceiver6_salmap': 586652265/1024/1024,  # 586.7 MB (586,652,265 bytes)
#     'perceiver6_salmap': 575300465/1024/1024,  # 575.3 MB (575,300,465 bytes)
#     # 'perceiver6': 57186921/1024/1024,  # 57.2 MB (57,186,921 bytes)
#     'perceiver6': 45835121/1024/1024,  # 45.8 MB (45,835,121 bytes)
#     # 45835121 / 575300465 = 0.0796
# }

# mn_dict = {
#     'Xu_CVPR': 'Xu(salmap)',
#     'Nguyen_MM': 'Ng(salmap)',
#     'TRACK': 'TR(salmap)',
#     'perceiver6': 'Ra(salxyz)',

#     'Xu_CVPR_salxyz': 'Xu(salxyz)',
#     'Nguyen_MM_salxyz': 'Ng(salxyz)',
#     'TRACK_salxyz': 'TR(salxyz)',
#     'perceiver6_salmap': 'Ra(salmap)',
# }

# # plt.bar([mn_dict[mn] for mn in model_size_dict.keys()], model_size_dict.values(), width=0.5, color=[color_dict[mn] for mn in model_size_dict.keys()])

# # 每个model的柱形图使用不同的底纹, 并设置底纹的间隔:
# hatch_str = '////'
# hatch_dict = {
#     'Xu_CVPR': '',
#     'Xu_CVPR_salxyz': hatch_str,
#     'Nguyen_MM': '',
#     'Nguyen_MM_salxyz': hatch_str,
#     'TRACK': '',
#     'TRACK_salxyz': hatch_str, 
#     'perceiver6_salmap': '',
#     'perceiver6': hatch_str,
# }
# for i, mn in enumerate(model_size_dict.keys()):
#     plt.bar([mn_dict[mn]], [model_size_dict[mn]], width=0.5, color=color_dict[mn], hatch=hatch_dict[mn])

# # 显示数据标签
# for a,b in zip([mn_dict[mn] for mn in model_size_dict.keys()], model_size_dict.values()):
#     plt.text(a,b, 
#             f'{b:.2f}',  # 保留两位小数:
#              ha='center', 
#              va='bottom',
#             )
    

# plt.ylabel('Model Size (MB)')
# # plt.xlabel('Algorithm')
# # 设定y轴的范围:
# plt.ylim(0, 600)
# plt.xticks(rotation=30)
# plt.grid()
# plot_path = os.path.join(f'{OUTPUT_FOLDER}/salxyz-model-size.pdf')
# plt.savefig(plot_path,
#             bbox_inches='tight',
#             transparent=True,
#             pad_inches=0)
# plt.clf()



# # endregion




# '''
# Part 7
# '''
# # region: 绘制将pos_only的orth-dist作为横轴从大到小排列时, Nguren_MM, Xu_CVPR, TRACK和perceiver6的orth-dist.
# mn_dict = {
#     'Reactive': 'static-baseline',
#     'LR': 'linear-regression',
#     'pos_only': 'deep-pos-only',
#     'Xu_CVPR': 'Xu_CVPR',
#     'Nguyen_MM': 'Nguyen_MM',
#     'TRACK': 'TRACK',
#     'perceiver6': 'Rainbow-VP',
# }
# plot_model_names = ['pos_only', 'Nguyen_MM', 'Xu_CVPR', 'TRACK', 'perceiver6']
# total_df = pd.DataFrame(columns=plot_model_names)
# for mn in plot_model_names:
#     df = pd.read_csv(f'./{dataset_name}/{mn}/Results_init_{INIT_WINDOW}_in_{M_WINDOW}_out_{H_WINDOW}_end_{END_WINDOW}/errors.csv')
#     # 按照video,user,x_i分组, 每组求平均error:
#     df = df.groupby(['video', 'user', 'x_i']).mean().reset_index()
#     total_df[mn] = df[met]

# # 将total_df按照'pos_only'列的值从小到大排序.
# total_df = total_df.sort_values(by=['pos_only'])
# # 每100行求平均:
# total_df = total_df.groupby(np.arange(len(total_df))//100).mean().reset_index()
# # 绘制图像: (以pos_only的值作为横轴, 其他列的值作为纵轴)
# for mn in plot_model_names[1:]:
#     plt.plot(total_df['pos_only'], total_df[mn], 
#              label=mn_dict[mn], color=color_dict[mn], 
#              linestyle=line_style_dict[mn], linewidth=2.0, 
#              marker=marker_dict[mn], markersize=5.0)


# # plt.title(f'Prediction Error ({dataset_name})')
# plt.ylabel('Orthodromic Distance of other models')
# plt.xlabel('Orthodromic Distance of deep-pos-only')
# # plt.ylim(0, 0.8)
# plt.legend()
# plt.grid()
# # plt.show()
# plot_path = os.path.join(f'{OUTPUT_FOLDER}/orth-dist2.pdf')
# plt.savefig(plot_path, 
#             bbox_inches='tight', 
#             transparent=True,
#             pad_inches=0)
# plt.clf()
# # endregion




# '''
# Part 8
# '''
# # region: 读取attention_probs文件中的所有矩阵, 并将他们相乘, 将最终结果画成热力图.
# attn_probs_folder = f'{OUTPUT_FOLDER}/attention_probs'
# attn_probs_files = os.listdir(attn_probs_folder)
# # 按照文件名中的整数从大到小排序:
# attn_probs_files = sorted(attn_probs_files, key=lambda x: int(x.split('.')[0]), reverse=True)

# def get_attn_probs(file_names):
#     # 读取所有矩阵, 并顺次相乘:
#     attn_probs = np.load(os.path.join(attn_probs_folder, file_names[0]))
#     # 对attn_probs在前两维求平均:
#     attn_probs = np.mean(attn_probs, axis=(0, 1))
#     for i in range(1, len(file_names)):
#         ap = np.load(os.path.join(attn_probs_folder, file_names[i]))
#         # 对ap在前两维求平均:
#         ap = np.mean(ap, axis=(0, 1))
#         attn_probs = np.matmul(attn_probs, ap)
#     # 对attn_probs每一行进行归一化:
#     attn_probs = attn_probs / np.sum(attn_probs, axis=1, keepdims=True)
#     return attn_probs

# # 每num个文件一组, 求一个attn_probs, 最后将所有attn_probs求平均:
# # num = 2*8+2
# # num = 2*32+2
# num = 1*1+2
# # num = 8*1+2

# attn_probs = get_attn_probs(attn_probs_files[:num])
# for i in tqdm(range(1, len(attn_probs_files)//num)):
#     attn_probs += get_attn_probs(attn_probs_files[i*num:(i+1)*num])
# attn_probs /= (len(attn_probs_files)//num)


# # 只保留后25行:
# attn_probs = attn_probs[-25:, :]

# # 在列上做归一化:
# # # 方式1:
# # max_value = np.max(attn_probs, axis=0, keepdims=True)
# # min_value = np.min(attn_probs, axis=0, keepdims=True)
# # attn_probs = (attn_probs - min_value) / (max_value - min_value)
# # 方式2:
# attn_probs = attn_probs / np.sum(attn_probs, axis=0, keepdims=True)

# print(attn_probs.shape)
# print(f'max: {np.max(attn_probs)}')
# print(f'min: {np.min(attn_probs)}')

# # 画shape=(25, 55)的attn_probs矩阵画成热力图:
# # 1. 对于前15列, 值从0到1, 由白色渐变为红色; 
# # 2. 对于后40列, 值从0到1, 由白色渐变为蓝色;


# # # region 画图:
# # gray_image = attn_probs  # shape=(25, 55)

# # def gray_to_color(gray_image, start_color, end_color):
# #     return (1 - gray_image)*start_color + gray_image*end_color

# # # 将颜色从十六进制转换为RGB值，然后再归一化到[0,1]
# # start_color = np.array([255, 255, 255])/255  # white color
# # end_color1 = np.array([248, 206, 204])/255  # #F8CECC
# # end_color2 = np.array([218, 232, 252])/255  # #DAE8FC

# # # 创建一个空的彩色图像
# # color_image = np.empty((25, 55, 3))

# # # 填充前15列
# # color_image[:, :15] = gray_to_color(gray_image[:, :15, np.newaxis], start_color, end_color1)
# # # 填充后40列
# # color_image[:, 15:] = gray_to_color(gray_image[:, 15:, np.newaxis], start_color, end_color2)

# # print(color_image.shape)
# # print(color_image[0, 0, :])
# # print(f'max: {np.max(color_image)}')
# # print(f'min: {np.min(color_image)}')

# # # 显示彩色图像
# # plt.imshow(color_image)
# # # endregion

# # plt.imshow(attn_probs, cmap='Greens')

# sns.heatmap(attn_probs, cmap='Blues', cbar=True, cbar_kws={'shrink': .53}, xticklabels=False, yticklabels=False, square=True, vmin=np.min(attn_probs), vmax=np.max(attn_probs))

# # plt.axis('off')
# plt.savefig(os.path.join(OUTPUT_FOLDER, 'attn_probs.pdf'),
#             bbox_inches='tight',
#             transparent=True,
#             pad_inches=0)
# plt.clf()



# # endregion




# '''
# Part 9
# '''
# # region: 绘制不同视频/用户的deep_pos_only和RainbowVP图
# mn_dict = {
#     'Reactive': 'static-baseline',
#     'LR': 'linear-regression',
#     'pos_only': 'deep-pos-only',
#     'Xu_CVPR': 'Xu_CVPR',
#     'Nguyen_MM': 'Nguyen_MM',
#     'TRACK': 'TRACK',
#     'perceiver6': 'Rainbow-VP',
# }
# plot_model_names = ['pos_only', 'perceiver6']
# df_p = pd.read_csv(f'./{dataset_name}/pos_only/Results_init_{INIT_WINDOW}_in_{M_WINDOW}_out_{H_WINDOW}_end_{END_WINDOW}/errors.csv')
# df_r = pd.read_csv(f'./{dataset_name}/perceiver6/Results_init_{INIT_WINDOW}_in_{M_WINDOW}_out_{H_WINDOW}_end_{END_WINDOW}/errors.csv')
# df_p = df_p.groupby(['video', 'user', 'x_i']).mean().reset_index()
# df_r = df_r.groupby(['video', 'user', 'x_i']).mean().reset_index()
# total_df = pd.DataFrame(columns=['video', 'user']+plot_model_names)
# total_df['video'] = df_p['video']
# total_df['user'] = df_p['user']
# total_df['pos_only'] = df_p[met]
# total_df['perceiver6'] = df_r[met]

# # 按照video进行分组, 并获取各个小df:
# for video in total_df['video'].unique():
#     df = total_df[total_df['video']==video]
#     # 将df按照pos_only的值从小到大排序:
#     df = df.sort_values(by=['pos_only'])[['pos_only', 'perceiver6']]
#     # # 每3行求平均:
#     # df = df.groupby(np.arange(len(df))//3).mean().reset_index()
#     # # 将df中每一列的值分别归一化到[0,1]:
#     # df = (df - df.min()) / (df.max() - df.min())

#     x = df['pos_only']
#     # # 方式1:
#     # y = df['perceiver6']
#     # 方式2:
#     from scipy import signal
#     y = signal.savgol_filter(df['perceiver6'], window_length=50, polyorder=2, mode='nearest')

#     # 绘制图像: (以pos_only的值作为横轴, RainbowVP的值作为纵轴)
#     plt.plot(x, y, 
#             label=video, #color=color_dict[mn], 
#             # linestyle=line_style_dict[mn], linewidth=2.0, 
#             # marker=marker_dict[mn], markersize=5.0
#             )
#     # 绘制一条黑色的y=x线:
#     plt.plot([0, 2], [0, 2], color='black', linestyle='--', linewidth=2.0)
#     plt.ylabel('Orthodromic Distance of Rainbow-VP')
#     plt.xlabel('Orthodromic Distance of deep-pos-only')
#     # 将label用两列显示, 并上移:
#     plt.legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, 2.0))
#     plt.grid()
#     # plt.show()
# plot_path = os.path.join(f'{OUTPUT_FOLDER}/video-direction.pdf')
# plt.savefig(plot_path, 
#             bbox_inches='tight', 
#             transparent=True,
#             pad_inches=0)
# plt.clf()


# # 按照user进行分组, 并获取各个小df:
# for user in total_df['user'].unique():
#     df = total_df[total_df['user']==user]
#     # 将df按照pos_only的值从小到大排序:
#     df = df.sort_values(by=['pos_only'])[['pos_only', 'perceiver6']]
#     # # 每3行求平均:
#     # df = df.groupby(np.arange(len(df))//3).mean().reset_index()
#     # # 将df中每一列的值分别归一化到[0,1]:
#     # df = (df - df.min()) / (df.max() - df.min())

#     x = df['pos_only']
#     # # 方式1:
#     # y = df['perceiver6']
#     # 方式2:
#     from scipy import signal
#     y = signal.savgol_filter(df['perceiver6'], window_length=50, polyorder=2, mode='nearest')

#     # 绘制图像: (以pos_only的值作为横轴, RainbowVP的值作为纵轴)
#     plt.plot(x, y, 
#             label=user, #color=color_dict[mn], 
#             # linestyle=line_style_dict[mn], linewidth=2.0, 
#             # marker=marker_dict[mn], markersize=5.0
#             )
#     # 绘制一条黑色的y=x线:
#     plt.plot([0, 2], [0, 2], color='black', linestyle='--', linewidth=2.0)
#     plt.ylabel('Orthodromic Distance of Rainbow-VP')
#     plt.xlabel('Orthodromic Distance of deep-pos-only')
#     # 将label用两列显示, 并上移:
#     plt.legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, 2.0))
#     plt.grid()
#     # plt.show()
# plot_path = os.path.join(f'{OUTPUT_FOLDER}/user-direction.pdf')
# plt.savefig(plot_path, 
#             bbox_inches='tight', 
#             transparent=True,
#             pad_inches=0)
# plt.clf()
# # endregion




# '''
# Part 10
# '''
# # region: 绘制不同视频/用户的视点移动距离分布图: 
# videos = get_video_ids(SAMPLED_DATASET_FOLDER)
# users = get_user_ids(SAMPLED_DATASET_FOLDER)
# users_per_video = get_users_per_video(SAMPLED_DATASET_FOLDER)
# all_traces = {}
# df_dist = pd.DataFrame(columns=['video', 'user', 'dist'])
# for video in videos:
#     all_traces[video] = {}
#     for user in users_per_video[video]:
#         all_traces[video][user] = read_sampled_positions_for_trace(SAMPLED_DATASET_FOLDER, str(video), str(user))  # np.ndarray, shape=(num_frames, 3)
#         # 计算dist: all_traces[video][user]每行的3个元素对应点的三维坐标, dist=两点之间的欧氏距离
#         dist = np.sqrt(np.sum(np.square(all_traces[video][user][1:] - all_traces[video][user][:-1]), axis=1))
#         # 将dist中的每一个元素, 都加上对应的video和user, 并添加到df_dist中 (不使用append):
#         df_dist = pd.concat([df_dist, pd.DataFrame({'video': [video]*len(dist), 'user': [user]*len(dist), 'dist': dist})], ignore_index=True)

# # 绘制不同video间的图像:
# # 将df_dist按照video进行分组, 并获取各个小df, 组成total_df:
# total_df = pd.DataFrame([])
# for video in df_dist['video'].unique():
#     total_df[video] = df_dist[df_dist['video']==video]['dist'].values

# # 绘制视点移动距离的核密度估计图:
# sns.displot(total_df, kind='kde', fill=False, palette='Set2', linewidth=1.1)
# plt.xlabel('Orthodromic Distance')
# plt.grid()
# plot_path = os.path.join(f'{OUTPUT_FOLDER}/video-speed-kde.pdf')
# plt.savefig(plot_path, 
#             bbox_inches='tight', 
#             transparent=True,
#             pad_inches=0)
# plt.clf()

# # 绘制视点移动距离的箱线图: 不绘制异常值, 并按照均值从小到大排序:
# # print(total_df.mean().sort_values().index)
# # exit()
# sns.boxplot(data=total_df, palette='Set2', linewidth=1.1, showfliers=False, order=total_df.mean().sort_values().index.tolist())
# plt.ylabel('Orthodromic Distance')
# plt.grid()
# plot_path = os.path.join(f'{OUTPUT_FOLDER}/video-speed-box.pdf')
# plt.savefig(plot_path, 
#             bbox_inches='tight', 
#             transparent=True,
#             pad_inches=0)
# plt.clf()

# # 绘制不同user间的图像:
# # 将df_dist按照user进行分组, 并获取各个小df, 组成total_df:
# total_df = pd.DataFrame([])
# for user in df_dist['user'].unique():
#     total_df[user] = df_dist[df_dist['user']==user]['dist'].values

# # 绘制视点移动距离的核密度估计图:
# sns.displot(total_df, kind='kde', fill=False, palette='Set2', linewidth=1.1)
# plt.xlabel('Orthodromic Distance')
# plt.grid()
# plot_path = os.path.join(f'{OUTPUT_FOLDER}/user-speed-kde.pdf')
# plt.savefig(plot_path, 
#             bbox_inches='tight', 
#             transparent=True,
#             pad_inches=0)
# plt.clf()

# # 绘制视点移动距离的箱线图: 不绘制异常值, 并按照均值从小到大排序:
# # print(total_df.mean().sort_values().index)
# # exit()
# sns.boxplot(data=total_df, palette='Set2', linewidth=1.1, showfliers=False, order=total_df.mean().sort_values().index.tolist())
# plt.ylabel('Orthodromic Distance')
# plt.grid()
# plot_path = os.path.join(f'{OUTPUT_FOLDER}/user-speed-box.pdf')
# plt.savefig(plot_path, 
#             bbox_inches='tight', 
#             transparent=True,
#             pad_inches=0)
# plt.clf()

# # endregion




# '''
# Part 11
# '''
# # region: 绘制不同模型和deep_pos_only预测误差之间的散点图
# mn_dict = {
#     'Reactive': 'static-baseline',
#     'LR': 'linear-regression',
#     'pos_only': 'deep-pos-only',
#     'Xu_CVPR': 'Xu_CVPR',
#     'Nguyen_MM': 'Nguyen_MM',
#     'TRACK': 'TRACK',
#     'perceiver6': 'Rainbow-VP',
# }
# plot_model_names = ['pos_only', 'Xu_CVPR', 'Nguyen_MM', 'TRACK', 'perceiver6']
# total_df = pd.DataFrame(columns=plot_model_names)
# for mn in plot_model_names:
#     df = pd.read_csv(f'./{dataset_name}/{mn}/Results_init_{INIT_WINDOW}_in_{M_WINDOW}_out_{H_WINDOW}_end_{END_WINDOW}/errors.csv')
#     # 按照video,user,x_i分组, 每组求平均error:
#     df = df.groupby(['video', 'user', 'x_i']).mean().reset_index()
#     total_df[mn] = df[met]

# # 将total_df按照'pos_only'列的值从小到大排序:
# total_df = total_df.sort_values(by=['pos_only'])
# # 以pos_only的值作为横轴, 其他列的值作为纵轴, 绘制散点图:
# for mn in plot_model_names[1:]:
#     # 散点图中需要包含拟合线和拟合区域:
#     sns.regplot(x='pos_only', y=mn, data=total_df, scatter=True, scatter_kws={'s': 6, 'alpha': 0.4}, line_kws={'color': color_dict[mn]}, label=mn_dict[mn], order=1)
    
    
# # 绘制一条黑色的y=x线:
# plt.plot([0, 2], [0, 2], color='black', linestyle='--', linewidth=2.0)
# plt.ylabel('Orthodromic Distance of other models')
# plt.xlabel('Orthodromic Distance of deep-pos-only')
# plt.legend()
# plt.grid()
# # plt.show()

# plot_path = os.path.join(f'{OUTPUT_FOLDER}/error-models-pos_only.pdf')
# plt.savefig(plot_path, 
#             bbox_inches='tight', 
#             transparent=True,
#             pad_inches=0)
# plt.clf()
# # endregion




# '''
# Part 12
# '''
# # region: 绘制不同视频/用户的模型误差分布图: 

# mn_dict = {
#     'Reactive': 'static-baseline',
#     'LR': 'linear-regression',
#     'pos_only': 'deep-pos-only',
#     'Xu_CVPR': 'Xu_CVPR',
#     'Nguyen_MM': 'Nguyen_MM',
#     'TRACK': 'TRACK',
#     'perceiver6': 'Rainbow-VP',
# }
# plot_model_names = ['pos_only', 'Xu_CVPR', 'Nguyen_MM', 'TRACK', 'perceiver6']
# for mn in plot_model_names:
#     df = pd.read_csv(f'./{dataset_name}/{mn}/Results_init_{INIT_WINDOW}_in_{M_WINDOW}_out_{H_WINDOW}_end_{END_WINDOW}/errors.csv')
#     # 按照video,user,x_i分组, 每组求平均error:
#     df = df.groupby(['video', 'user', 'x_i']).mean().reset_index()
#     # 绘制当前的mn模型对每个video预测误差的核密度估计图:
#     sns.displot(df, x=met, hue='video', kind='kde', fill=False, palette='Set2', linewidth=1.1)
#     plt.xlabel('Orthodromic Distance')
#     plt.grid()
#     plot_path = os.path.join(f'{OUTPUT_FOLDER}/error-video-{mn}-kde.pdf')
#     plt.savefig(plot_path, 
#                 bbox_inches='tight', 
#                 transparent=True,
#                 pad_inches=0)
#     plt.clf()

#     # 绘制当前的mn模型对每个user预测误差的核密度估计图:
#     sns.displot(df, x=met, hue='user', kind='kde', fill=False, palette='Set2', linewidth=1.1)
#     plt.xlabel('Orthodromic Distance')
#     plt.grid()
#     plot_path = os.path.join(f'{OUTPUT_FOLDER}/error-user-{mn}-kde.pdf')
#     plt.savefig(plot_path, 
#                 bbox_inches='tight', 
#                 transparent=True,
#                 pad_inches=0)
#     plt.clf()

# # endregion




# '''
# Part 13
# '''
# # region: 为每个video-user单独绘制orth-dist-{video}-{user}.pdf

# mn_dict = {
#     'Reactive': 'static-baseline',
#     'LR': 'linear-regression',
#     'pos_only': 'deep-pos-only',
#     'Xu_CVPR': 'Xu_CVPR',
#     'Nguyen_MM': 'Nguyen_MM',
#     'TRACK': 'TRACK',
#     'perceiver6': 'Rainbow-VP (ours)',
    
#     'perceiver6_salmap': 'Rainbow-VP-salmap',
#     'perceiver6_motmap': 'Rainbow-VP-motmap',
#     'perceiver6_motxyz': 'Rainbow-VP-motxyz',
    
#     'perceiver5': 'perceiver5',
# }
# plot_model_names = ['pos_only', 'Nguyen_MM', 'Xu_CVPR', 'TRACK', 'perceiver6']
# # 通过读取plot_model_names中的第一个model的errors.csv文件, 获取video和user的信息:
# df = pd.read_csv(f'./{dataset_name}/{plot_model_names[0]}/Results_init_{INIT_WINDOW}_in_{M_WINDOW}_out_{H_WINDOW}_end_{END_WINDOW}/errors.csv')
# videos = df['video'].unique()
# users = df['user'].unique()
# for video in videos:
#     for user in users:
#         for mn in plot_model_names:
#             df = pd.read_csv(f'./{dataset_name}/{mn}/Results_init_{INIT_WINDOW}_in_{M_WINDOW}_out_{H_WINDOW}_end_{END_WINDOW}/errors.csv')
#             avg_error_per_timestep = []
#             for t in range(H_WINDOW):
#                 error = df[(df['t']==t) & (df['video']==video) & (df['user']==user)][met].mean()
#                 avg_error_per_timestep.append(error)
#             plt.plot(np.arange(1, H_WINDOW+1)*RATE, avg_error_per_timestep, 
#                     label=mn_dict[mn], color=color_dict[mn], 
#                     linestyle=line_style_dict[mn], linewidth=2.0, 
#                     marker=marker_dict[mn], markevery=3, markersize=5.0)
#         # plt.title(f'Prediction Error ({dataset_name})')
#         plt.ylabel('Avg. Orthodromic Distance')
#         plt.xlabel('Prediction step (sec.)')
#         # plt.ylim(0, 0.8)
#         # 自定义labels的位置:
#         plt.legend(loc='lower right', bbox_to_anchor=(1.1, 0.0))
#         plt.grid()
#         # plt.show()
#         plot_path = os.path.join(f'{OUTPUT_FOLDER}/orth-dist-{video}-{user}.pdf')
#         plt.savefig(plot_path, 
#                     bbox_inches='tight', 
#                     transparent=True,
#                     pad_inches=0)
#         plt.clf()
#         plt.close()
# # endregion




# '''
# Part 14
# '''
# # region: 绘制trajectories:

# def get_trajectories_folder(model_name):
#     return f'./plot_trajectories_figures/{dataset_name}/{model_name}/init_{INIT_WINDOW}_in_{M_WINDOW}_out_{H_WINDOW}_end_{END_WINDOW}'

# def get_trajectories_df(model_name, video, user, x_i):
#     file_path = f'{get_trajectories_folder(model_name)}/video_{video}_user_{user}_x_i_{x_i}.csv'
#     df = pd.read_csv(file_path)
#     return df

# def draw_trajectory(theta, phi, color, label, flag_draw_labels):
#     x = theta.values / np.pi
#     y = 1 - phi.values / np.pi
#     # plt.plot(x, y, color=color, label=label, linewidth=1)#, marker='o', markevery=3, markersize=5.0, markerfacecolor='none')
#     if flag_draw_labels:
#         plt.arrow(x[0], y[0], x[1]-x[0], y[1]-y[0], width=0.003, length_includes_head=True, head_width=0.02, head_length=0.03, color=color, label=label)
#     for i in range(1 if flag_draw_labels else 0, len(x)-1):
#         diff_x = x[i+1]-x[i]
#         if abs(diff_x+2) < abs(diff_x):
#             diff_x += 2
#         elif abs(diff_x-2) < abs(diff_x):
#             diff_x -= 2
#         diff_y = y[i+1]-y[i]
#         plt.arrow(x[i], y[i], diff_x, diff_y, width=0.001, length_includes_head=True, head_width=0.01, head_length=0.02, color=color)
#     # plt.arrow(x[-2], y[-2], x[-1]-x[-2], y[-1]-y[-2], width=0.0001, length_includes_head=True, head_width=0.005, head_length=0.01, color=color)

# mn_dict = {
#     'Reactive': 'static-baseline',
#     'LR': 'linear-regression',
#     'pos_only': 'deep-pos-only',
#     'Xu_CVPR': 'Xu_CVPR',
#     'Nguyen_MM': 'Nguyen_MM',
#     'TRACK': 'TRACK',
#     'perceiver6': 'Rainbow-VP',
# }
# single_model_names = ['pos_only', 'Xu_CVPR', 'Nguyen_MM', 'TRACK', 'perceiver6']
# double_model_names = [['TRACK', 'pos_only'], ['perceiver6', 'pos_only']]


# def get_saliency_map(video, x_i, t=0):
#     saliency_folder = os.path.join('./', dataset_name, 'saliency_paver_big')
#     salmaps_path = os.path.join(saliency_folder, video + '.npy')
#     salmaps = np.load(salmaps_path)
#     salmap = salmaps[t+x_i]
    
#     # region: old
#     # # 创建透明度数组
#     # # alpha = np.ones_like(salmap)
#     # # alpha = np.sin(salmap)
#     # alpha = salmap
#     # # alpha[salmap < 0.4] = 0

#     # # 正向
#     # # cmap_name = 'hot'
#     # # cmap_name = 'summer'
#     # cmap_name = 'afmhot'

#     # # 反向
#     # salmap = 1 - salmap
#     # cmap_name = 'YlGnBu'

#     # cmap = plt.colormaps.get_cmap(cmap_name)
#     # salmap = cmap(salmap)

#     # # 设置透明度
#     # salmap[..., -1] = alpha

#     # return salmap
#     # endregion

#     return salmap


# # 对单模型轨迹进行绘制:
# flag_draw_labels = False
# for mn in single_model_names:
#     bests, worsts = get_bests_worsts_vuxi(dataset_name, mn, INIT_WINDOW, M_WINDOW, H_WINDOW, END_WINDOW)
#     for i, vuxi in enumerate(bests+worsts):
#         video, user, x_i = vuxi
#         print(f'Generating {video}-{user}-{x_i}...')

#         # 绘制salmap:
#         salmap = get_saliency_map(video, x_i)
#         plt.imshow(salmap, extent=[0, 2, 0, 1], origin='lower', alpha=0.5, zorder=1) 

#         # 绘制轨迹:
#         df = get_trajectories_df(mn, video, user, x_i)
#         draw_trajectory(df['theta'][M_WINDOW+H_WINDOW : M_WINDOW+H_WINDOW+H_WINDOW], df['phi'][M_WINDOW+H_WINDOW : M_WINDOW+H_WINDOW+H_WINDOW], 
#                         color_dict[mn], f'prediction of {mn_dict[mn]}', flag_draw_labels)
#         draw_trajectory(df['theta'][:M_WINDOW], df['phi'][:M_WINDOW], 
#                         'grey', 'historical viewpoint trajectory (HT)', flag_draw_labels)
#         draw_trajectory(df['theta'][M_WINDOW:M_WINDOW+H_WINDOW], df['phi'][M_WINDOW:M_WINDOW+H_WINDOW], 
#                         'black', 'ground-truth future trajectory', flag_draw_labels)

#         # 设定x轴和y轴的范围:
#         plt.xlim(0, 2)
#         plt.ylim(0, 1)
#         plt.gca().set_aspect(1)
#         plt.axis('off')
        
#         if flag_draw_labels:
#             plt.legend(ncol=1)
        
#         output_folder = os.path.join(OUTPUT_FOLDER, 'trajectories', mn, 'bests' if vuxi in bests else 'worsts')
#         os.makedirs(output_folder, exist_ok=True)
#         plot_path = os.path.join(f'{output_folder}/({i}) {video}-{user}-{x_i}.pdf')
#         plt.savefig(plot_path,
#                     bbox_inches='tight',
#                     transparent=True,
#                     pad_inches=0)
#         plt.clf()
#         plt.close()


# # 对多模型轨迹进行绘制:
# flag_draw_labels = False
# for mns in double_model_names:
#     mn0, mn1 = mns
#     bests, worsts = get_bests_worsts_vuxi(dataset_name, [mn0, mn1], INIT_WINDOW, M_WINDOW, H_WINDOW, END_WINDOW)
#     for i, vuxi in enumerate(bests+worsts):
#         video, user, x_i = vuxi
#         print(f'Generating {video}-{user}-{x_i}...')

#         # 绘制salmap:
#         salmap = get_saliency_map(video, x_i)
#         plt.imshow(salmap, extent=[0, 2, 0, 1], origin='lower', alpha=0.5, zorder=1) 

#         # 绘制轨迹:
#         df0 = get_trajectories_df(mn0, video, user, x_i)
#         df1 = get_trajectories_df(mn1, video, user, x_i)
#         draw_trajectory(df0['theta'][M_WINDOW+H_WINDOW : M_WINDOW+H_WINDOW+H_WINDOW], df['phi'][M_WINDOW+H_WINDOW : M_WINDOW+H_WINDOW+H_WINDOW], 
#                         color_dict[mn0], f'prediction of {mn_dict[mn0]}', flag_draw_labels)
#         draw_trajectory(df0['theta'][:M_WINDOW], df['phi'][:M_WINDOW], 
#                         'grey', 'historical viewpoint trajectory (HT)', flag_draw_labels)
#         draw_trajectory(df0['theta'][M_WINDOW:M_WINDOW+H_WINDOW], df['phi'][M_WINDOW:M_WINDOW+H_WINDOW], 
#                         'black', 'ground-truth future trajectory', flag_draw_labels)
#         draw_trajectory(df1['theta'][M_WINDOW+H_WINDOW : M_WINDOW+H_WINDOW+H_WINDOW], df['phi'][M_WINDOW+H_WINDOW : M_WINDOW+H_WINDOW+H_WINDOW],
#                         color_dict[mn1], f'prediction of {mn_dict[mn1]}', flag_draw_labels)

#         # 设定x轴和y轴的范围:
#         plt.xlim(0, 2)
#         plt.ylim(0, 1)
#         plt.gca().set_aspect(1)
#         plt.axis('off')
        
#         if flag_draw_labels:
#             plt.legend(ncol=1)
        
#         output_folder = os.path.join(OUTPUT_FOLDER, 'trajectories', f'{mn0}-{mn1}', 'bests' if vuxi in bests else 'worsts')
#         os.makedirs(output_folder, exist_ok=True)
#         plot_path = os.path.join(f'{output_folder}/({i}) {video}-{user}-{x_i}.pdf')
#         plt.savefig(plot_path,
#                     bbox_inches='tight',
#                     transparent=True,
#                     pad_inches=0)
#         plt.clf()
#         plt.close()
# # endregion




# '''
# Part 15
# '''
# # region: 绘制error_perceicer6预测出的orth-dist和perceiver6实际的orth-dist之间的散点图:
# mn_dict = {
#     'perceiver6': 'Rainbow-VP',
#     'error_perceiver6': 'Error-Rainbow-VP',
# }
# plot_model_names = ['perceiver6', 'error_perceiver6']
# total_df = pd.DataFrame(columns=plot_model_names)
# for mn in plot_model_names:
#     df = pd.read_csv(f'./{dataset_name}/{mn}/Results_init_{INIT_WINDOW}_in_{M_WINDOW}_out_{H_WINDOW}_end_{END_WINDOW}/errors.csv')
#     total_df[mn] = df[met]

# # 将total_df按照'perceiver6'列的值从小到大排序:
# total_df = total_df.sort_values(by=['perceiver6'])
# # 以'perceiver6'的值作为横轴, 其它列 ('error_perceiver') 的值作为纵轴, 绘制散点图:
# for mn in plot_model_names[1:]:
#     # 散点图中需要包含拟合线和拟合区域:
#     sns.regplot(x='perceiver6', y=mn, data=total_df, scatter=True, scatter_kws={'s': 1, 'alpha': 0.2}, label=mn_dict[mn], fit_reg=False)
    
# # 设置x轴和y轴的显示范围均为[0, 3.5]
# plt.xlim(0, 3.5)
# plt.ylim(0, 3.5)
# # 设置x轴和y轴的刻度间隔
# x_ticks = np.arange(0, 3.5, 0.5)
# y_ticks = np.arange(0, 3.5, 0.5)
# # 设置x轴和y轴的刻度位置和间隔
# plt.xticks(x_ticks)
# plt.yticks(y_ticks)
# # 设置x轴和y轴的刻度长度相等
# plt.axis('equal')

# plt.ylabel('Orthodromic Distance of Error-Rainbow-VP')
# plt.xlabel('Orthodromic Distance of Rainbow-VP')
# # plt.legend()
# plt.grid()
# # plt.show()

# plot_path = os.path.join(f'{OUTPUT_FOLDER}/error-models-perceiver6.pdf')
# plt.savefig(plot_path, 
#             bbox_inches='tight', 
#             transparent=True,
#             pad_inches=0)
# plt.clf()
# # endregion




# '''
# Part 16
# '''
# # region: 绘制error_perceiver6预测出的perceiver6的误差分布图, 以及perceiver6实际的误差分布图: 

# mn_dict = {
#     'Reactive': 'static-baseline',
#     'LR': 'linear-regression',
#     'pos_only': 'deep-pos-only',
#     'Xu_CVPR': 'Xu_CVPR',
#     'Nguyen_MM': 'Nguyen_MM',
#     'TRACK': 'TRACK',
#     'perceiver6': 'Rainbow-VP',
#     'error_perceiver6': 'Error-Rainbow-VP',
# }
# plot_model_names = ['perceiver6', 'error_perceiver6']
# for mn in plot_model_names:
#     df = pd.read_csv(f'./{dataset_name}/{mn}/Results_init_{INIT_WINDOW}_in_{M_WINDOW}_out_{H_WINDOW}_end_{END_WINDOW}/errors.csv')
#     sns.displot(df, x=met, kind='kde', fill=False, linewidth=1.1)
#     plt.xlabel(f'Orthodromic Distance of {mn_dict[mn]}')
#     plt.grid()
#     plot_path = os.path.join(f'{OUTPUT_FOLDER}/error-kde-{mn}.pdf')
#     plt.savefig(plot_path, 
#                 bbox_inches='tight', 
#                 transparent=True,
#                 pad_inches=0)
#     plt.clf()

# # endregion




# '''
# Part 17
# '''
# # region: 绘制不同的tile划分粒度下, use cross-tile-boundary-loss和不use cross-tile-boundary-loss的模型的预测误差之间的对比图:

# mn_dict = {
#     'Reactive': 'static-baseline',
#     'LR': 'linear-regression',
#     'pos_only': 'deep-pos-only',
#     'Xu_CVPR': 'Xu_CVPR',
#     'Nguyen_MM': 'Nguyen_MM',
#     'TRACK': 'TRACK',
#     'perceiver6': 'Rainbow-VP (ours)',
    
#     'perceiver6_salmap': 'Rainbow-VP-salmap',
#     'perceiver6_motmap': 'Rainbow-VP-motmap',
#     'perceiver6_motxyz': 'Rainbow-VP-motxyz',
    
#     'perceiver5': 'perceiver5',

#     'error_perceiver6': 'Eror-Rainbow-VP',
# }
# plot_model_names = ['perceiver6']
# plot_tilemap_shapes = [(i, 2*i) for i in range(2, 11)]

# for mn in plot_model_names:
#     for (tile_h, tile_w) in plot_tilemap_shapes:
#         dfs = []
#         dfs.append(pd.read_csv(f'./{dataset_name}/{mn}/Results_init_{INIT_WINDOW}_in_{M_WINDOW}_out_{H_WINDOW}_end_{END_WINDOW}_{tile_h}x{tile_w}/errors.csv'))
#         dfs.append(pd.read_csv(f'./{dataset_name}/{mn}/Results_init_{INIT_WINDOW}_in_{M_WINDOW}_out_{H_WINDOW}_end_{END_WINDOW}_{tile_h}x{tile_w}_useCtbLoss/errors.csv'))
#         colors = ['blue', 'red']
#         labels = ['without CtbLoss', 'with CtbLoss']

#         for i, df in enumerate(dfs):
#             avg_error_per_timestep = []
#             for t in range(H_WINDOW):
#                 error = df[df['t']==t][met].mean()
#                 avg_error_per_timestep.append(error)
#             # 输出当前model 2s-5s (15 time steps) 的平均error:
#             print(f'{mn_dict[mn]}: {np.mean(avg_error_per_timestep[-15:])}')
#             plt.plot(np.arange(1, H_WINDOW+1)*RATE, avg_error_per_timestep, 
#                     label=labels[i], color=colors[i],)

#         # plt.title(f'Prediction Error ({dataset_name})')
#         plt.ylabel('Avg. Manhattan Distance')
#         plt.xlabel('Prediction step (sec.)')
#         # plt.ylim(0, 0.8)
#         # 自定义labels的位置:
#         plt.legend(loc='lower right', bbox_to_anchor=(1.1, 0.0))
#         plt.grid()
#         # plt.show()
#         plot_path = os.path.join(f'{OUTPUT_FOLDER}/manh-dist-{mn}-{tile_h}x{tile_w}.pdf')
#         plt.savefig(plot_path, 
#                     bbox_inches='tight', 
#                     transparent=True,
#                     pad_inches=0)
#         plt.clf()
# # endregion




# '''
# Part 18
# '''
# # region: 绘制之前5s的预测误差和下5s的预测误差之间的关系;
# mn_dict = {
#     'perceiver6': 'Rainbow-VP',
# }
# mn = 'perceiver6'
# df = pd.read_csv(f'./{dataset_name}/{mn}/Results_init_{INIT_WINDOW}_in_{M_WINDOW}_out_{H_WINDOW}_end_{END_WINDOW}/errors.csv')
# total_df = pd.DataFrame(columns=['x', 'y'])
# videos = df['video'].unique()
# users = df['user'].unique()
# for video in videos:
#     for user in users:
#         # 取出当前video-user的df:
#         df_vu = df[(df['video']==video) & (df['user']==user)][[met]]
#         # 每5行求平均, 得到每一秒的预测误差:
#         df_vu = df_vu.groupby(np.arange(len(df_vu))//5).mean().reset_index()
#         # 将df_vu[:-5]作为dfx, 将_df[5:]作为dfy, 并且重置index:
#         dfx, dfy = df_vu[:-5].reset_index(), df_vu[5:].reset_index()
#         # 将dfx[met]连接到total_df的'x'列上, 将dfy连接到total_df的'y'列上:
#         total_df = pd.concat([total_df, pd.DataFrame({'x': dfx[met], 'y': dfy[met]})], ignore_index=True)

# print(total_df.tail(30))

# # 散点图中需要包含拟合线和拟合区域:
# sns.regplot(x='x', y='y', data=total_df, scatter=True, scatter_kws={'s': 1, 'alpha': 0.2})
    
# # 设置x轴和y轴的显示范围均为[0, 3.5]
# plt.xlim(0, 3.5)
# plt.ylim(0, 3.5)
# # 设置x轴和y轴的刻度间隔
# x_ticks = np.arange(0, 3.5, 0.5)
# y_ticks = np.arange(0, 3.5, 0.5)
# # 设置x轴和y轴的刻度位置和间隔
# plt.xticks(x_ticks)
# plt.yticks(y_ticks)
# # 设置x轴和y轴的刻度长度相等
# plt.axis('equal')

# plt.ylabel('Post 5s')
# plt.xlabel('Pre 5s')
# # plt.legend()
# plt.grid()
# # plt.show()

# plot_path = os.path.join(f'{OUTPUT_FOLDER}/error-pre_post.pdf')
# plt.savefig(plot_path, 
#             bbox_inches='tight', 
#             transparent=True,
#             pad_inches=0)
# plt.clf()
# # endregion




# '''
# Part 19
# '''
# # region: 绘制第4s的预测误差和第5s的预测误差之间的关系;
# mn_dict = {
#     'perceiver6': 'Rainbow-VP',
# }
# mn = 'perceiver6'
# df = pd.read_csv(f'./{dataset_name}/{mn}/Results_init_{INIT_WINDOW}_in_{M_WINDOW}_out_{H_WINDOW}_end_{END_WINDOW}/errors.csv')
# total_df = pd.DataFrame(columns=['x', 'y'])
# videos = df['video'].unique()
# users = df['user'].unique()
# for video in videos:
#     for user in users:
#         # 取出当前video-user的df:
#         df_vu = df[(df['video']==video) & (df['user']==user)][[met]]
#         # 每5行求平均, 得到每一秒的预测误差:
#         df_vu = df_vu.groupby(np.arange(len(df_vu))//5).mean().reset_index()
#         # 将df_vu中index对5取余为3的行作为dfx, 将df_vu中index对5取余为4的行作为dfy, 并且重置index:
#         dfx, dfy = df_vu[df_vu['index']%5==1].reset_index(), df_vu[df_vu['index']%5==4].reset_index()
#         # 将dfx[met]连接到total_df的'x'列上, 将dfy连接到total_df的'y'列上:
#         total_df = pd.concat([total_df, pd.DataFrame({'x': dfx[met], 'y': dfy[met]})], ignore_index=True)

# # 散点图中需要包含拟合线和拟合区域:
# sns.regplot(x='x', y='y', data=total_df, scatter=True, scatter_kws={'s': 2, 'alpha': 0.3})

# # 绘制y=x的直线:
# plt.plot([0, 3.5], [0, 3.5], color='black', linestyle='--', linewidth=2.0)
# # 设置x轴和y轴的显示范围均为[0, 3.5]
# plt.xlim(0, 3.5)
# plt.ylim(0, 3.5)
# # 设置x轴和y轴的刻度间隔
# x_ticks = np.arange(0, 3.5, 0.5)
# y_ticks = np.arange(0, 3.5, 0.5)
# # 设置x轴和y轴的刻度位置和间隔
# plt.xticks(x_ticks)
# plt.yticks(y_ticks)
# # 设置x轴和y轴的刻度长度相等
# plt.axis('equal')

# plt.ylabel('5s')
# plt.xlabel('4s')
# # plt.legend()
# plt.grid()
# # plt.show()

# plot_path = os.path.join(f'{OUTPUT_FOLDER}/error-4s-5s.pdf')
# plt.savefig(plot_path, 
#             bbox_inches='tight', 
#             transparent=True,
#             pad_inches=0)
# plt.clf()
# # endregion



