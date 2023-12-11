import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import argparse
import sys
sys.path.append('./')

parser = argparse.ArgumentParser(description='Plot the models error.')

parser.add_argument('--dataset_name', action='store', dest='dataset_name', help='The name of the dataset used to train this network.')
parser.add_argument('--init_window', action='store', dest='init_window', help='(Optional) Initial buffer window (to avoid stationary part).', type=int)
parser.add_argument('--m_window', action='store', dest='m_window', help='Past history window.', type=int)
parser.add_argument('--h_window', action='store', dest='h_window', help='Prediction window.', type=int)
parser.add_argument('--end_window', action='store', dest='end_window', help='(Optional) Final buffer (to avoid having samples with less outputs).', type=int)

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


save_folder = f'./plot_loss_hist/{dataset_name}/init_{INIT_WINDOW}_in_{M_WINDOW}_out_{H_WINDOW}_end_{END_WINDOW}'
os.makedirs(save_folder, exist_ok=True)

# 使用seaborn, 将3个model的df中orthodromic_distance列的hist图/cdf图绘制到一张图中:
plot_model_names = ['pos_only', 'TRACK', 'perceiver6']
colors = ['r', 'g', 'b']
for i in range(len(plot_model_names)):
    model_name = plot_model_names[i]
    color = colors[i]
    df = pd.read_csv(f'./{dataset_name}/{model_name}/Results_init_{INIT_WINDOW}_in_{M_WINDOW}_out_{H_WINDOW}_end_{END_WINDOW}/errors.csv')
    # 绘制hist图
    sns.distplot(df['orthodromic_distance'], hist=False, kde=True, rug=False, color=color, label=model_name, kde_kws={'linewidth': 0.5})
    # 绘制cdf图
    sns.kdeplot(df['orthodromic_distance'], cumulative=True, color=color, label=model_name, linewidth=0.5)

plt.title('orthodromic_distance on %s dataset' % (dataset_name))
plt.ylabel('Density')
plt.xlabel('orthodromic_distance')
plt.legend()
plt.grid()

plot_path = os.path.join(f'{save_folder}/orthodromic_distance.png')
plt.savefig(plot_path)
plt.clf()