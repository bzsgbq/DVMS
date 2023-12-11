import numpy as np
import torch

torch.backends.cudnn.enabled = False  # 这是为了解决运行含有LSTM模块的模型时会报错的问题; 代价是模型的运行速度变慢: RuntimeError: cuDNN error: CUDNN_STATUS_INTERNAL_ERROR

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import os
import pandas as pd
import pickle
import sys
from tqdm.auto import tqdm
import time
import argparse

sys.path.insert(0, './')

from utils.load_dataset import DatasetLoader
from utils.save_output import Tee
from utils.utils import *



# default parameters
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
EPOCHS = 200
ROTATE = False  # 如果置为True, 则会将视点数据绕z轴旋转随机角度, 旨在增强训练出的模型的泛化性; 但是实际测试后发现效果不明显; (该功能只适用于不需要用到salmap的模型)
INC_M = True  # 是否考虑也让perceiver模型将前m_window个视点位置预测出来参与loss计算;

num_feat_dim = 3


parser = argparse.ArgumentParser(description='Process the input parameters to train the network.')
parser.add_argument('--train', action="store_true", dest='train_flag',
                    help='Train the model.')
parser.add_argument('--evaluate', action="store_true", dest='evaluate_flag',
                    help='Save error and output trajectories on test split of the dataset.')
parser.add_argument('--gpu_id', action='store', dest='gpu_id', help='The gpu used to train this network.')
parser.add_argument('--dataset_name', action='store', dest='dataset_name',
                    help='The name of the dataset used to train this network.')
parser.add_argument('--model_name', action='store', dest='model_name',
                    help='The name of the model used to reference the network structure used.')
parser.add_argument('--train_weights', action='store', dest='train_weights',
                    help='(Optional) Path to model weights to train from instead of training from scratch.')
parser.add_argument('--evaluate_model_dir', action='store', dest='evaluate_model_dir',
                    help='(Optional) Path to the model to be evaluated.')
parser.add_argument('--evaluate_output_dir', action='store', dest='evaluate_output_dir',
                    help='(Optional) Path to save the error file of the evaluated model.')
parser.add_argument('--init_window', action='store', dest='init_window',
                    help='(Optional) Initial buffer window (to avoid stationary part).', type=int)
parser.add_argument('--m_window', action='store', dest='m_window', help='Past history window.', type=int)
parser.add_argument('--h_window', action='store', dest='h_window', help='Prediction window.', type=int)
parser.add_argument('--end_window', action='store', dest='end_window',
                    help='(Optional) Final buffer (to avoid having samples with less outputs).', type=int)
parser.add_argument('--provided_videos', action="store_true", dest='provided_videos',
                    help='Flag that tells whether the list of videos is provided in a global variable.')
parser.add_argument('--use_true_saliency', action="store_true", dest='use_true_saliency',
                    help='Flag that tells whether the true saliency is used.')
parser.add_argument('--lr', action="store", dest='lr', help='(Optional) Neural network learning rate.')
parser.add_argument('--bs', action="store", dest='bs', help='(Optional) Neural network batch size.')
parser.add_argument('--epochs', action="store", dest='epochs', help='(Optional) Number of epochs.')

parser.add_argument('--tile_h', action="store", dest='tile_h', help='(Optional) Tile map height.', type=int)
parser.add_argument('--tile_w', action="store", dest='tile_w', help='(Optional) Tile map width.', type=int)
parser.add_argument('--use_cross_tile_boundary_loss', action="store_true", dest='use_cross_tile_boundary_loss',
                    help='Flag that tells whether the "cross tile boundary loss" is used.')
args = parser.parse_args()


M_WINDOW = args.m_window
H_WINDOW = args.h_window

if args.gpu_id is None:
    device = torch.device('cpu')
    print('WARNING: No GPU selected.')
else:
    try:
        device = torch.device(f'cuda:{int(args.gpu_id)}')
    except TypeError:
        device = torch.device('cpu')
        print('WARNING: No GPU selected.')

if args.init_window is None:
    INIT_WINDOW = M_WINDOW
else:
    INIT_WINDOW = args.init_window

if args.end_window is None:
    END_WINDOW = H_WINDOW
else:
    END_WINDOW = args.end_window

if args.bs is not None:
    BATCH_SIZE = int(args.bs)

if args.lr is not None:
    LEARNING_RATE = float(args.lr)

if args.epochs is not None:
    EPOCHS = int(args.epochs)

TRAIN_MODEL = False
EVALUATE_MODEL = False

if args.train_flag:
    TRAIN_MODEL = True
if args.evaluate_flag:
    EVALUATE_MODEL = True

TILEMAP_SHAPE = (args.tile_h, args.tile_w) if args.tile_h is not None and args.tile_w is not None else None


dataset_name = args.dataset_name
assert dataset_name in ['David_MMSys_18', 'Wu_MMSys_17', 'Xu_CVPR_18']
model_name = args.model_name
assert model_name in ['pos_only', 'pos_only_plus', 'pos_only_single', 'TRACK', 'TRACK_plus', 'TRACK_res', 'TRACK_convlstm', 'TRACK_posal', 
                      'TRACK_ablat_sal', 'TRACK_ablat_fuse', 'TRACK_ablat_all', 'TRACK_left', 'TRACK_sage', 'TRACK_mask', 'TRACK_deform_conv', 
                      'informer', 'informer_plus', 'mts_mixer', 'pos_only_fuse', 'pos_only_fuse2', 'convlstm', 'gpt4', 'sage',
                      'VideoMAE', 'perceiver', 'perceiver2', 'perceiver3', 'perceiver4', 'perceiver5', 'perceiver6', 'perceiver7', 'perceiver8', 'perceiver9', 'perceiver10',
                      'Xu_CVPR', 'Nguyen_MM', 'Reactive', 'LR', 
                      'Xu_CVPR_salxyz', 'Nguyen_MM_salxyz', 'TRACK_salxyz', 'perceiver6_salmap', 'perceiver6_motmap', 'perceiver6_motxyz', 
                      'RainbowVP', 'RainbowVP_salmap', 'RainbowVP_motmap', 
                      'error_perceiver6']
# 将model_name中'RainbowVP'的部分替换为'perceiver6':
model_name = model_name.replace('RainbowVP', 'perceiver6')


# datasets and dataloaders
dataset_loader = DatasetLoader(dataset_name, model_name, M_WINDOW, H_WINDOW, INIT_WINDOW, END_WINDOW, args.provided_videos, args.use_true_saliency, args.tile_h, args.tile_w, args.use_cross_tile_boundary_loss)
train_dataset, val_dataset, test_dataset = dataset_loader.load()
print(f'len(train_dataset) = {len(train_dataset)}')
print(f'len(val_dataset) = {len(val_dataset)}')
print(f'len(test_dataset) = {len(test_dataset)}')


# models
if model_name == 'pos_only':
    from models.pos_only import PosOnlyModel
    model = PosOnlyModel(M_WINDOW=M_WINDOW, H_WINDOW=H_WINDOW).to(device)
elif model_name == 'pos_only_plus':
    from models.pos_only_plus import PosOnlyPlusModel
    model = PosOnlyPlusModel(M_WINDOW=M_WINDOW, H_WINDOW=H_WINDOW).to(device)
elif model_name == 'pos_only_single':
    from models.pos_only_single import PosOnlySingleModel
    model = PosOnlySingleModel(M_WINDOW=M_WINDOW, H_WINDOW=H_WINDOW).to(device)
elif model_name == 'Reactive':
    from models.reactive import ReactiveModel
    model = ReactiveModel(M_WINDOW=M_WINDOW, H_WINDOW=H_WINDOW).to(device)
elif model_name == 'LR':
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()

elif model_name in ['TRACK', 'TRACK_salxyz']:
    from models.TRACK import TRACKModel
    model = TRACKModel(M_WINDOW=M_WINDOW, H_WINDOW=H_WINDOW, salmap_shape=dataset_loader.salmap_shape).to(device)
elif model_name in ['Xu_CVPR', 'Xu_CVPR_salxyz']:
    from models.cvpr18 import CVPR18Model
    model = CVPR18Model(M_WINDOW=M_WINDOW, H_WINDOW=H_WINDOW, salmap_shape=dataset_loader.salmap_shape).to(device)
elif model_name in ['Nguyen_MM', 'Nguyen_MM_salxyz']:
    from models.mm18 import MM18Model
    model = MM18Model(M_WINDOW=M_WINDOW, H_WINDOW=H_WINDOW, salmap_shape=dataset_loader.salmap_shape).to(device)

elif model_name == 'TRACK_plus':
    from models.TRACK_plus import TRACKPlusModel
    model = TRACKPlusModel(M_WINDOW=M_WINDOW, H_WINDOW=H_WINDOW, salmap_shape=dataset_loader.salmap_shape).to(device)
elif model_name == 'TRACK_res':
    from models.TRACK_res import TRACKResModel
    model = TRACKResModel(M_WINDOW=M_WINDOW, H_WINDOW=H_WINDOW, salmap_shape=dataset_loader.salmap_shape).to(device)
elif model_name == 'TRACK_convlstm':
    from models.TRACK_convlstm import TRACKConvLstmModel
    model = TRACKConvLstmModel(M_WINDOW=M_WINDOW, H_WINDOW=H_WINDOW, salmap_shape=dataset_loader.salmap_shape).to(device)
elif model_name == 'TRACK_ablat_sal':
    from models.TRACK_ablat_sal import TRACKAblatSalModel
    model = TRACKAblatSalModel(M_WINDOW=M_WINDOW, H_WINDOW=H_WINDOW, salmap_shape=dataset_loader.salmap_shape).to(device)
elif model_name == 'TRACK_ablat_fuse':
    from models.TRACK_ablat_fuse import TRACKAblatFuseModel
    model = TRACKAblatFuseModel(M_WINDOW=M_WINDOW, H_WINDOW=H_WINDOW, salmap_shape=dataset_loader.salmap_shape).to(device)
elif model_name == 'TRACK_ablat_all':
    from models.TRACK_ablat_all import TRACKAblatAllModel
    model = TRACKAblatAllModel(M_WINDOW=M_WINDOW, H_WINDOW=H_WINDOW, salmap_shape=dataset_loader.salmap_shape).to(device)
elif model_name == 'TRACK_left':
    from models.TRACK_left import TRACKLeftModel
    model = TRACKLeftModel(M_WINDOW=M_WINDOW, H_WINDOW=H_WINDOW, salmap_shape=dataset_loader.salmap_shape).to(device)
elif model_name == 'TRACK_sage':
    from models.TRACK_sage import TRACKSageModel
    model = TRACKSageModel(M_WINDOW=M_WINDOW, H_WINDOW=H_WINDOW, salmap_shape=dataset_loader.salmap_shape).to(device)
elif model_name == 'TRACK_mask':
    from models.TRACK_mask import TRACKMaskModel
    model = TRACKMaskModel(M_WINDOW=M_WINDOW, H_WINDOW=H_WINDOW, salmap_shape=dataset_loader.salmap_shape).to(device)
elif model_name == 'TRACK_deform_conv':
    from models.TRACK_deform_conv import TRACKDeformConvModel
    model = TRACKDeformConvModel(M_WINDOW=M_WINDOW, H_WINDOW=H_WINDOW, salmap_shape=dataset_loader.salmap_shape).to(device)
elif model_name == 'TRACK_posal':
    from models.TRACK_posal import TRACKPosalModel
    model = TRACKPosalModel(M_WINDOW=M_WINDOW, H_WINDOW=H_WINDOW, salmap_shape=dataset_loader.salmap_shape).to(device)

elif model_name == 'informer':
    from models.Informer.model import create_informer_model
    from models.Informer.configs import configs
    configs.seq_len = M_WINDOW
    configs.label_len = M_WINDOW
    configs.pred_len = H_WINDOW
    configs.d_model = 128
    configs.d_ff = 256
    configs.dropout = 0.5
    model = create_informer_model(configs).to(device)
elif model_name == 'informer_plus':
    from models.informer_plus import InformerPlusModel
    model = InformerPlusModel(M_WINDOW=M_WINDOW, H_WINDOW=H_WINDOW, salmap_shape=dataset_loader.salmap_shape).to(device)

elif model_name == 'mts_mixer':
    from models.MTSMixer.model import MTSMixer
    from models.MTSMixer.configs import configs
    configs.seq_len = M_WINDOW
    configs.pred_len = H_WINDOW
    model = MTSMixer(configs).to(device)
elif model_name == 'pos_only_fuse':
    from models.pos_only_fuse import PosOnlyFuseModel
    model = PosOnlyFuseModel(M_WINDOW=M_WINDOW, H_WINDOW=H_WINDOW, tilemap_shape=TILEMAP_SHAPE, salmap_shape=dataset_loader.salmap_shape).to(device)
elif model_name == 'pos_only_fuse2':
    from models.pos_only import PosOnlyModel
    from models.fuse import FusePosSal
    pos_model = PosOnlyModel(M_WINDOW=M_WINDOW, H_WINDOW=H_WINDOW).to(device)
    pos_model_dir = os.path.join('./', dataset_name, 'pos_only', 'Models' + f'_init_{INIT_WINDOW}_in_{M_WINDOW}_out_{H_WINDOW}_end_{END_WINDOW}')
    pos_model.load_state_dict(torch.load(pos_model_dir + '/weights.pth', map_location=device))
    model = FusePosSal(H_WINDOW=H_WINDOW, tilemap_shape=TILEMAP_SHAPE, salmap_shape=dataset_loader.salmap_shape).to(device)
elif model_name == 'convlstm':
    from models.convlstm import ConvLSTM_model
    model = ConvLSTM_model(M_WINDOW=M_WINDOW, H_WINDOW=H_WINDOW, tilemap_shape=TILEMAP_SHAPE, salmap_shape=dataset_loader.salmap_shape).to(device)
elif model_name == 'gpt4':
    from models.gpt4 import ViewpointPredictor
    model = ViewpointPredictor(H_WINDOW=H_WINDOW, salmap_shape=dataset_loader.salmap_shape).to(device)
elif model_name == 'sage':
    from models.sage import ViewpointPredictionModel
    model = ViewpointPredictionModel(M_WINDOW=M_WINDOW, H_WINDOW=H_WINDOW, salmap_shape=dataset_loader.salmap_shape).to(device)

elif model_name == 'VideoMAE':
    from models.videomae_vp import VideoMAEForViewpointPrediction
    model_ckpt = "MCG-NJU/videomae-base-finetuned-kinetics"
    model = VideoMAEForViewpointPrediction.from_pretrained(model_ckpt, ignore_mismatched_sizes=True).to(device)

elif model_name == 'perceiver':
    from models.perceiver_vp import PerceiverVPModel
    from transformers import PerceiverConfig
    cfg = PerceiverConfig(
        num_latents=64,
        d_latents=512,
        num_blocks=4,#1,
        num_self_attends_per_block=4,#26,
        num_self_attention_heads=8,
        num_cross_attention_heads=8,
        qk_channels=None,
        v_channels=None,
        cross_attention_shape_for_attention="kv",
        self_attention_widening_factor=1,
        cross_attention_widening_factor=1,
        hidden_act="gelu",
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        use_query_residual=True,
    )
    model = PerceiverVPModel(config=cfg, M_WINDOW=M_WINDOW, H_WINDOW=H_WINDOW, salmap_shape=dataset_loader.salmap_shape, hidden_size=511).to(device)

elif model_name == 'perceiver2':
    from models.perceiver_vp2 import PerceiverVPModel2
    from transformers import PerceiverConfig
    cfg = PerceiverConfig(
        num_latents=256,#256,
        d_latents=256,
        num_blocks=2,#2,#1,
        num_self_attends_per_block=8,#6,#26,
        num_self_attention_heads=8,
        num_cross_attention_heads=8,
        qk_channels=None,
        v_channels=None,
        cross_attention_shape_for_attention="kv",
        self_attention_widening_factor=1,
        cross_attention_widening_factor=1,
        hidden_act="gelu",
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        use_query_residual=True,
    )
    model = PerceiverVPModel2(config=cfg, M_WINDOW=M_WINDOW, H_WINDOW=H_WINDOW, salmap_shape=dataset_loader.salmap_shape, pos_dim=3).to(device)

elif model_name == 'perceiver3':
    from models.perceiver_vp3 import PerceiverVPModel3
    from transformers import PerceiverConfig
    cfg = PerceiverConfig(
        num_latents=256,
        d_latents=256,
        num_blocks=2,#1,
        num_self_attends_per_block=8,#26,
        num_self_attention_heads=8,
        num_cross_attention_heads=8,
        qk_channels=None,
        v_channels=None,
        cross_attention_shape_for_attention="kv",
        self_attention_widening_factor=1,
        cross_attention_widening_factor=1,
        hidden_act="gelu",
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        use_query_residual=True,
    )
    model = PerceiverVPModel3(config=cfg, M_WINDOW=M_WINDOW, H_WINDOW=H_WINDOW, salmap_shape=dataset_loader.salmap_shape, pos_dim=3).to(device)

elif model_name == 'perceiver4':
    from models.perceiver_vp4 import PerceiverVPModel4
    from transformers import PerceiverConfig
    cfg = PerceiverConfig(
        num_latents=256,#256,
        d_latents=256,
        num_blocks=2,#2,#1,
        num_self_attends_per_block=8,#6,#26,
        num_self_attention_heads=8,
        num_cross_attention_heads=8,
        qk_channels=None,
        v_channels=None,
        cross_attention_shape_for_attention="kv",
        self_attention_widening_factor=1,
        cross_attention_widening_factor=1,
        hidden_act="gelu",
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        use_query_residual=True,
    )
    model = PerceiverVPModel4(config=cfg, M_WINDOW=M_WINDOW, H_WINDOW=H_WINDOW, salmap_shape=dataset_loader.salmap_shape, device=device).to(device)

elif model_name == 'perceiver5':
    from models.perceiver_vp5 import PerceiverVPModel5
    from transformers import PerceiverConfig
    cfg = PerceiverConfig(
        num_latents=40,#256,
        d_latents=256,
        num_blocks=2,#2,#1,
        num_self_attends_per_block=6,#6,#26,
        num_self_attention_heads=8,
        num_cross_attention_heads=8,
        qk_channels=None,
        v_channels=None,
        cross_attention_shape_for_attention="kv",
        self_attention_widening_factor=1,
        cross_attention_widening_factor=1,
        hidden_act="gelu",
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        use_query_residual=True,
    )
    model = PerceiverVPModel5(config=cfg, M_WINDOW=M_WINDOW, H_WINDOW=H_WINDOW).to(device)

elif model_name == 'perceiver6':
    from models.perceiver_vp6 import PerceiverVPModel6
    from transformers import PerceiverConfig
    cfg = PerceiverConfig(
        num_latents=40,#40,
        d_latents=256,
        num_blocks=1,#1,
        num_self_attends_per_block=1,#1,
        num_self_attention_heads=8,
        num_cross_attention_heads=8,
        qk_channels=None,
        v_channels=None,
        cross_attention_shape_for_attention="kv",
        self_attention_widening_factor=1,
        cross_attention_widening_factor=1,
        hidden_act="gelu",
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        use_query_residual=True,
    )
    model = PerceiverVPModel6(config=cfg, M_WINDOW=M_WINDOW, H_WINDOW=H_WINDOW, salmap_shape=dataset_loader.salmap_shape).to(device)

elif model_name == 'perceiver6_salmap':
    from models.perceiver_vp6_salmap import PerceiverVPSalmapModel6
    from transformers import PerceiverConfig
    cfg = PerceiverConfig(
        num_latents=40,
        d_latents=256,
        num_blocks=1,
        num_self_attends_per_block=1,
        num_self_attention_heads=8,
        num_cross_attention_heads=8,
        qk_channels=None,
        v_channels=None,
        cross_attention_shape_for_attention="kv",
        self_attention_widening_factor=1,
        cross_attention_widening_factor=1,
        hidden_act="gelu",
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        use_query_residual=True,
    )
    model = PerceiverVPSalmapModel6(config=cfg, M_WINDOW=M_WINDOW, H_WINDOW=H_WINDOW, salmap_shape=dataset_loader.salmap_shape).to(device)

elif model_name == 'perceiver6_motmap':
    from models.perceiver_vp6_motmap import PerceiverVPMotmapModel6
    from transformers import PerceiverConfig
    cfg = PerceiverConfig(
        num_latents=40,
        d_latents=256,
        num_blocks=1,
        num_self_attends_per_block=1,
        num_self_attention_heads=8,
        num_cross_attention_heads=8,
        qk_channels=None,
        v_channels=None,
        cross_attention_shape_for_attention="kv",
        self_attention_widening_factor=1,
        cross_attention_widening_factor=1,
        hidden_act="gelu",
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        use_query_residual=True,
    )
    model = PerceiverVPMotmapModel6(config=cfg, M_WINDOW=M_WINDOW, H_WINDOW=H_WINDOW, motmap_shape=dataset_loader.motmap_shape).to(device)

elif model_name == 'perceiver6_motxyz':
    from models.perceiver_vp6_motxyz import PerceiverVPMotxyzModel6
    from transformers import PerceiverConfig
    cfg = PerceiverConfig(
        num_latents=40,
        d_latents=256,
        num_blocks=1,
        num_self_attends_per_block=1,
        num_self_attention_heads=8,
        num_cross_attention_heads=8,
        qk_channels=None,
        v_channels=None,
        cross_attention_shape_for_attention="kv",
        self_attention_widening_factor=1,
        cross_attention_widening_factor=1,
        hidden_act="gelu",
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        use_query_residual=True,
    )
    model = PerceiverVPMotxyzModel6(config=cfg, M_WINDOW=M_WINDOW, H_WINDOW=H_WINDOW, motmap_shape=dataset_loader.motmap_shape).to(device)

elif model_name == 'perceiver7':
    from models.perceiver_vp7 import PerceiverVPModel7
    from transformers import PerceiverConfig
    cfg = PerceiverConfig(
        num_latents=512,
        d_latents=2048,
        num_blocks=1,#2,#1,
        num_self_attends_per_block=1,#6,#26,
        num_self_attention_heads=8,
        num_cross_attention_heads=8,
        qk_channels=None,
        v_channels=None,
        cross_attention_shape_for_attention="kv",
        self_attention_widening_factor=1,
        cross_attention_widening_factor=1,
        hidden_act="gelu",
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        use_query_residual=True,
    )
    model = PerceiverVPModel7(config=cfg, M_WINDOW=M_WINDOW, H_WINDOW=H_WINDOW, salmap_shape=dataset_loader.salmap_shape).to(device)

elif model_name == 'perceiver8':
    from models.perceiver_vp8 import PerceiverVPModel8
    from transformers import PerceiverConfig
    cfg = PerceiverConfig(
        num_latents=256,#256,
        d_latents=256,
        num_blocks=2,#2,#1,
        num_self_attends_per_block=8,#6,#26,
        num_self_attention_heads=8,
        num_cross_attention_heads=8,
        qk_channels=None,
        v_channels=None,
        cross_attention_shape_for_attention="kv",
        self_attention_widening_factor=1,
        cross_attention_widening_factor=1,
        hidden_act="gelu",
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        use_query_residual=True,
    )
    model = PerceiverVPModel8(config=cfg, M_WINDOW=M_WINDOW, H_WINDOW=H_WINDOW, salmap_shape=dataset_loader.salmap_shape).to(device)

elif model_name == 'perceiver9':
    from models.perceiver_vp9 import PerceiverVPModel9
    from transformers import PerceiverConfig
    cfg = PerceiverConfig(
        num_latents=256,#256,
        d_latents=256,
        num_blocks=2,#2,#1,
        num_self_attends_per_block=8,#6,#26,
        num_self_attention_heads=8,
        num_cross_attention_heads=8,
        qk_channels=None,
        v_channels=None,
        cross_attention_shape_for_attention="kv",
        self_attention_widening_factor=1,
        cross_attention_widening_factor=1,
        hidden_act="gelu",
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        use_query_residual=True,
    )
    model = PerceiverVPModel9(config=cfg, M_WINDOW=M_WINDOW, H_WINDOW=H_WINDOW, salmap_shape=dataset_loader.salmap_shape).to(device)

elif model_name == 'perceiver10':
    from models.perceiver_vp10 import PerceiverVPModel10
    from transformers import PerceiverConfig
    cfg = PerceiverConfig(
        num_latents=256,#256,
        d_latents=256,
        num_blocks=2,#2,#1,
        num_self_attends_per_block=8,#6,#26,
        num_self_attention_heads=8,
        num_cross_attention_heads=8,
        qk_channels=None,
        v_channels=None,
        cross_attention_shape_for_attention="kv",
        self_attention_widening_factor=1,
        cross_attention_widening_factor=1,
        hidden_act="gelu",
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        use_query_residual=True,
    )
    model = PerceiverVPModel10(config=cfg, M_WINDOW=M_WINDOW, H_WINDOW=H_WINDOW, salmap_shape=dataset_loader.salmap_shape).to(device)

elif model_name == 'error_perceiver6':
    from models.error_perceiver6 import ErrorPerceiver6Model
    from models.perceiver_vp6 import PerceiverVPModel6
    from transformers import PerceiverConfig
    error_cfg = PerceiverConfig(
        num_latents=40,#40,
        d_latents=256,
        num_blocks=1,#1,
        num_self_attends_per_block=1,#1,
        num_self_attention_heads=8,
        num_cross_attention_heads=8,
        qk_channels=None,
        v_channels=None,
        cross_attention_shape_for_attention="kv",
        self_attention_widening_factor=1,
        cross_attention_widening_factor=1,
        hidden_act="gelu",
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        use_query_residual=True,
    )
    cfg = PerceiverConfig(
        num_latents=40,#40,
        d_latents=256,
        num_blocks=1,#1,
        num_self_attends_per_block=1,#1,
        num_self_attention_heads=8,
        num_cross_attention_heads=8,
        qk_channels=None,
        v_channels=None,
        cross_attention_shape_for_attention="kv",
        self_attention_widening_factor=1,
        cross_attention_widening_factor=1,
        hidden_act="gelu",
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        use_query_residual=True,
    )
    model = ErrorPerceiver6Model(config=error_cfg, M_WINDOW=M_WINDOW, H_WINDOW=H_WINDOW, salmap_shape=dataset_loader.salmap_shape).to(device)
    perceiver_model = PerceiverVPModel6(config=cfg, M_WINDOW=M_WINDOW, H_WINDOW=H_WINDOW, salmap_shape=dataset_loader.salmap_shape).to(device)

elif model_name == 'error_TRACK':
    from models.perceiver_vp6 import PerceiverVPModel6
    from transformers import PerceiverConfig
    cfg = PerceiverConfig(
        num_latents=40,#40,
        d_latents=256,
        num_blocks=1,#1,
        num_self_attends_per_block=1,#1,
        num_self_attention_heads=8,
        num_cross_attention_heads=8,
        qk_channels=None,
        v_channels=None,
        cross_attention_shape_for_attention="kv",
        self_attention_widening_factor=1,
        cross_attention_widening_factor=1,
        hidden_act="gelu",
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        use_query_residual=True,
    )
    perceiver_model = PerceiverVPModel6(config=cfg, M_WINDOW=M_WINDOW, H_WINDOW=H_WINDOW, salmap_shape=dataset_loader.salmap_shape).to(device)
    from models.error_TRACK import ErrorTRACKModel
    model = ErrorTRACKModel(M_WINDOW=M_WINDOW, H_WINDOW=H_WINDOW, salmap_shape=dataset_loader.salmap_shape).to(device)
else:
    raise NotImplementedError(f'Model {model_name} not implemented yet.')

if args.use_cross_tile_boundary_loss:
    assert args.tile_h is not None and args.tile_w is not None
    loss_func = lambda true, pred : all_metrics['mse_ctb'](true, pred, tilemap_shape=TILEMAP_SHAPE)
else:
    loss_func = lambda true, pred : all_metrics['mse'](true, pred, tilemap_shape=TILEMAP_SHAPE)

get_orth_dist = all_metrics['orth']
get_manh_dist = lambda true, pred : all_metrics['manh'](true, pred, tilemap_shape=TILEMAP_SHAPE)
get_acc_prec_reca_f1 = lambda true, pred : all_metrics['aprf'](true, pred, tilemap_shape=TILEMAP_SHAPE)


if TRAIN_MODEL:
    os.makedirs(dataset_loader.models_folder, exist_ok=True)
    os.makedirs(dataset_loader.results_folder, exist_ok=True)

    if args.train_weights is not None:
        model.load_state_dict(torch.load(args.train_weights + '/weights.pth', map_location=device))

    output_log = open(dataset_loader.results_folder + '/console.log', 'w')
    sys.stdout = Tee(sys.__stdout__, output_log)
    
    if model_name == 'LR':
        print(f'Loading {dataset_name} dataset...')
        train_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True, pin_memory=True)
        print('Loading finished.')
        for batch, (X, y, video, user, x_i) in enumerate(train_dataloader):   
            X, y = X.numpy(), y.numpy()
            batch_size = y.shape[0]
            X, y = X.reshape(batch_size, -1), y.reshape(batch_size, -1)
            model.fit(X, y)
        pickle.dump(model, open(dataset_loader.models_folder + '/model.pkl', 'wb'))
        print(f'Training finished.')
        exit(0)

    print(f'Loading {dataset_name} dataset...')
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    print('Loading finished.')

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    # if model_name == 'error_perceiver6':
    #     scheduler = MultiStepLR(optimizer, milestones=[15, 20], gamma=0.1, verbose=True)
    # else:
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5, min_lr=1e-7, verbose=True)
    
    print(f'Training {model_name} on {dataset_name} - Batch size: {BATCH_SIZE} - Learning rate: {LEARNING_RATE}')
    
    if model_name == 'error_perceiver6':
        for epoch in range(EPOCHS):
            print(f"Epoch {epoch + 1}/{EPOCHS}    ", end='')
            # TRAINING: 
            model.train()
            model_dir = dataset_loader.models_folder if args.evaluate_model_dir is None else args.evaluate_model_dir
            perceiver_model_dir = model_dir.replace('error_perceiver6', 'perceiver6')
            perceiver_model.load_state_dict(torch.load(perceiver_model_dir + '/weights.pth', map_location=device))
            perceiver_model.eval()

            train_loss_lst = []
            val_loss_lst = []

            for batch, (X, y, video, user, x_i) in enumerate(val_dataloader):  # NOTE: 对于error_perceiver6来说, 它通过perceiver6在val set上的预测误差进行训练;
                if isinstance(X, list):
                    X, y = [x.float().to(device) for x in X], y.float().to(device)
                elif isinstance(X, torch.Tensor):
                    X, y = X.float().to(device), y.float().to(device)
                else:
                    raise NotImplementedError('X must be a list or a tensor.')

                batch_size = y.shape[0]
                optimizer.zero_grad()

                # Compute prediction and loss
                if model_name == 'pos_only_fuse2':
                    X = [pos_model(X[:2]), X[2]]
                if dataset_name == 'Wu_MMSys_17':
                    y = y[..., -2:] if num_feat_dim == 2 else y[..., :3]
                if 'perceiver' in model_name:
                    enc_pos_in = X[0][..., -2:] if num_feat_dim == 2 else X[0][..., :3]
                    y = torch.cat([enc_pos_in, y], dim=1) if INC_M else y
                
                error_pred = model(X)
                with torch.no_grad():
                    y_pred = perceiver_model(X)
                    error_true = get_orth_dist(y, y_pred)  # (batch_size, h_window, 1)
                loss = loss_func(error_true, error_pred)  # (batch_size, h_window, 1)
                
                # Backpropagation
                loss.backward()
                optimizer.step()

                batch_train_loss = torch.mean(loss.detach()).item()
                train_loss_lst.append(batch_train_loss)
            
            else:
                # Print average training loss and metrics
                train_loss = np.mean(train_loss_lst)
                print(f"train_loss: {train_loss},    ", end='')

                # VALIDATION: 
                model.eval()
                with torch.no_grad():
                    for batch, (X, y, video, user, x_i) in enumerate(test_dataloader):  # NOTE: 对于error_perceiver6来说, 它将perceiver6的测试集作为验证集;
                        if isinstance(X, list):
                            X, y = [x.float().to(device) for x in X], y.float().to(device)
                        elif isinstance(X, torch.Tensor):
                            X, y = X.float().to(device), y.float().to(device)
                        else:
                            raise NotImplementedError('X must be a list or a tensor.')

                        batch_size = y.shape[0]
                        # Compute prediction and loss
                        if model_name == 'pos_only_fuse2':
                            X = [pos_model(X[:2]), X[2]]
                        if dataset_name == 'Wu_MMSys_17':
                            y = y[..., -2:] if num_feat_dim == 2 else y[..., :3]
                        if 'perceiver' in model_name:
                            enc_pos_in = X[0][..., -2:] if num_feat_dim == 2 else X[0][..., :3]
                            y = torch.cat([enc_pos_in, y], dim=1) if INC_M else y
                        
                        error_pred = model(X)
                        with torch.no_grad():
                            y_pred = perceiver_model(X)
                            error_true = get_orth_dist(y, y_pred)  # (batch_size, h_window, 1)
                        loss = loss_func(error_true, error_pred)  # (batch_size, h_window, 1)
                        batch_val_loss = torch.mean(loss.detach()).item()
                        val_loss_lst.append(batch_val_loss)
                val_loss = np.mean(val_loss_lst)
                print(f"val_loss: {np.mean(val_loss):>8f}")
                scheduler.step(val_loss)
                # scheduler.step()

                if epoch == 0:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), dataset_loader.models_folder + '/weights.pth')
                else:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(model.state_dict(), dataset_loader.models_folder + '/weights.pth')
                continue
            
            print('Training has been interrupted.')
            break

        print(f'Training finished.')
        exit(0)

    # 其余模型的训练过程:
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}    ", end='')
        # TRAINING
        model.train()
        train_loss_lst = []
        val_loss_lst = []

        for batch, (X, y, video, user, x_i) in enumerate(train_dataloader):
            if isinstance(X, list):
                X, y = [x.float().to(device) for x in X], y.float().to(device)
            elif isinstance(X, torch.Tensor):
                X, y = X.float().to(device), y.float().to(device)
            else:
                raise NotImplementedError('X must be a list or a tensor.')

            if ROTATE:
                from utils.load_dataset import MODELS_USING_SALIENCY
                assert model_name not in MODELS_USING_SALIENCY
                # 这里对每个视点在单位球上的三维坐标, 绕z轴作任意角度的随机旋转; 本质上是一种数据增强, 旨在增强模型的泛化性.
                np.random.seed(7)  # Fixing random state for reproducibility
                random_degree = 360.0 * np.random.rand()
                if isinstance(X, list):
                    X, y = [data_rotation(x, random_degree) for x in X], data_rotation(y, random_degree)
                elif isinstance(X, torch.Tensor):
                    X, y = data_rotation(X, random_degree), data_rotation(y, random_degree)
                else:
                    raise NotImplementedError('X must be a list or a tensor.')
            
            batch_size = y.shape[0]
            optimizer.zero_grad()

            # Compute prediction and loss
            if model_name == 'pos_only_fuse2':
                X = [pos_model(X[:2]), X[2]]
            if dataset_name == 'Wu_MMSys_17':
                y = y[..., -2:] if num_feat_dim == 2 else y[..., :3]
            if 'perceiver' in model_name:
                enc_pos_in = X[0][..., -2:] if num_feat_dim == 2 else X[0][..., :3]
                y = torch.cat([enc_pos_in, y], dim=1) if INC_M else y
            y_pred = model(X)

            if torch.isnan(y_pred).any():
                print('`\nNan detected in model output!')
                break
            loss = loss_func(y, y_pred)  # (batch_size, h_window, 1)
            
            # Backpropagation
            loss.backward()
            optimizer.step()

            batch_train_loss = torch.mean(loss.detach()).item()
            train_loss_lst.append(batch_train_loss)

        else:
            # Print average training loss and metrics
            train_loss = np.mean(train_loss_lst)
            print(f"train_loss: {train_loss},    ", end='')

            # VALIDATION
            model.eval()
            with torch.no_grad():
                for batch, (X, y, video, user, x_i) in enumerate(val_dataloader):
                    if isinstance(X, list):
                        X, y = [x.float().to(device) for x in X], y.float().to(device)
                    elif isinstance(X, torch.Tensor):
                        X, y = X.float().to(device), y.float().to(device)
                    else:
                        raise NotImplementedError('X must be a list or a tensor.')
                    
                    batch_size = y.shape[0]
                    if model_name == 'pos_only_fuse2':
                        X = [pos_model(X[:2]), X[2]]
                    if dataset_name == 'Wu_MMSys_17':
                        y = y[..., -2:] if num_feat_dim == 2 else y[..., :3]
                    if 'perceiver' in model_name:
                        enc_pos_in = X[0][..., -2:] if num_feat_dim == 2 else X[0][..., :3]
                        y = torch.cat([enc_pos_in, y], dim=1) if INC_M else y
                    y_pred = model(X)
                    loss = loss_func(y, y_pred)  # (batch_size, h_window, 1)
                    batch_val_loss = torch.mean(loss.detach()).item()
                    val_loss_lst.append(batch_val_loss)
            val_loss = np.mean(val_loss_lst)
            print(f"val_loss: {np.mean(val_loss):>8f}")
            scheduler.step(val_loss)

            if epoch == 0:
                best_val_loss = val_loss
                torch.save(model.state_dict(), dataset_loader.models_folder + '/weights.pth')
            else:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), dataset_loader.models_folder + '/weights.pth')
            continue
        
        print('Training has been interrupted.')
        break

    print(f'Training finished.')

if EVALUATE_MODEL:
    if model_name == 'LR':
        print(f'Loading {dataset_name} dataset...')
        test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, pin_memory=True)
        print('Loading finished.')
        
        model_dir = dataset_loader.models_folder if args.evaluate_model_dir is None else args.evaluate_model_dir
        output_dir = dataset_loader.results_folder if args.evaluate_output_dir is None else args.evaluate_output_dir
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        print(f'Loading {model_name} model...')
        model = pickle.load(open(model_dir + '/model.pkl', 'rb'))
        print('Loading finished.')

        errors = []
        running_time = 0
        print(f'Evaluating {model_name} on {dataset_name}...')
        # TEST
        for batch, (X, y, video, user, x_i) in enumerate(test_dataloader):
            X, y = X.numpy(), y.numpy()
            batch_size = y.shape[0]
            t = time.time()
            y_pred = model.predict(X.reshape(batch_size, -1)).reshape(batch_size, H_WINDOW, -1)
            running_time += time.time() - t
            y, y_pred = torch.from_numpy(y).float().to(device), torch.from_numpy(y_pred).float().to(device)
            orth_dist = get_orth_dist(y, y_pred)  # (batch_size, h_window, 1)
            manh_dist = get_manh_dist(y, y_pred)  # manhattan distance
            acc, prec, reca, f1_score = get_acc_prec_reca_f1(y, y_pred)  # (batch_size, h_window, 1)
            for i in range(batch_size):
                for t in range(H_WINDOW):
                    errors.append({'video': video[i], 'user': user[i], 'x_i': x_i[i].cpu().item(), 't': t, 
                                   'orthodromic_distance': orth_dist[i][t][0].cpu().item(),
                                   'manhattan_distance': manh_dist[i][t][0].cpu().item(),
                                   'accuracy': acc[i][t][0].cpu().item(),
                                   'precision': prec[i][t][0].cpu().item(),
                                   'recall': reca[i][t][0].cpu().item(),
                                   'f1_score': f1_score[i][t][0].cpu().item(),
                                   })

        pickle.dump(errors, open(output_dir + f'/errors.pkl', 'wb'))
        print(f'Evaluation Finished.')
        print(f"{model_name}'s Avg. Running Time: {1000*running_time / len(test_dataloader)} ms")

        df = pd.DataFrame.from_records(errors)
        df.to_csv(output_dir + f'/errors.csv', index=False)
        print(f'Results saved in {output_dir}.')
        exit(0)

    with torch.no_grad():
        print(f'Loading {dataset_name} dataset...')
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
        print('Loading finished.')
        
        model_dir = dataset_loader.models_folder if args.evaluate_model_dir is None else args.evaluate_model_dir
        output_dir = dataset_loader.results_folder if args.evaluate_output_dir is None else args.evaluate_output_dir
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        print(f'Loading {model_name} model...')
        model.load_state_dict(torch.load(model_dir + '/weights.pth', map_location=device))
        print('Loading finished.')

        errors = []
        model.eval()

        if model_name == 'error_perceiver6':
            perceiver_model_dir = model_dir.replace('error_perceiver6', 'perceiver6')
            perceiver_model.load_state_dict(torch.load(perceiver_model_dir + '/weights.pth', map_location=device))
            perceiver_model.eval()

        print(f'Evaluating {model_name} on {dataset_name}')
        running_time = 0
        for batch, (X, y, video, user, x_i) in enumerate(tqdm(test_dataloader)):
            if isinstance(X, list):
                X, y = [x.float().to(device) for x in X], y.float().to(device)
            elif isinstance(X, torch.Tensor):
                X, y = X.float().to(device), y.float().to(device)
            else:
                raise NotImplementedError('X must be a list or a tensor.')
            batch_size = y.shape[0]
            if model_name == 'pos_only_fuse2':
                X = [pos_model(X[:2]), X[2]]
            if dataset_name == 'Wu_MMSys_17':
                y = y[..., :3]
            t = time.time()

            if model_name == 'error_perceiver6':
                error_pred = model.predict(X)
                running_time += time.time() - t
                y_pred = perceiver_model.predict(X)
                error_true = get_orth_dist(y, y_pred)  # (batch_size, h_window, 1)

                # # 方式1: 将error_pred和error_true之间的差距作为orth_dist:
                # orth_dist = torch.abs(error_true - error_pred)  # (batch_size, h_window, 1)
                # 方式2: 直接将error_pred作为orth_dist:
                orth_dist = error_pred  # (batch_size, h_window, 1)

                for i in range(batch_size):
                    for t in range(H_WINDOW):
                        errors.append({'video': video[i], 'user': user[i], 'x_i': x_i[i].cpu().item(), 't': t, 
                                    'orthodromic_distance': orth_dist[i][t][0].cpu().item(),
                                    })
                continue

            y_pred = model.predict(X)
            running_time += time.time() - t
            
            # # 直观展示预测结果
            # # region: part 1
            # losses = [loss_func(y[i, -H_WINDOW:, :], y_pred[i]).cpu().detach().numpy() for i in range(y.shape[0])]
            # # 取loss最小的5个样本的index以及loss最大的5个样本的index
            # min_loss_idx = np.argsort(losses)[:5]
            # max_loss_idx = np.argsort(losses)[-5:]
            # # 以10为步长取从小到大的loss的index
            # sparse_loss_idx = np.argsort(losses)[::10]

            # save_folder = f'./plot_trajectories_figures/{dataset_name}/{model_name}/init_{INIT_WINDOW}_in_{M_WINDOW}_out_{H_WINDOW}_end_{END_WINDOW}'
            # os.makedirs(save_folder, exist_ok=True)
            # for i in range(len(min_loss_idx)):
            #     idx = min_loss_idx[i]
            #     plot_trajectories(save_path=f'{save_folder}/min_{i}_{idx}.png', 
            #                       x=X[0][idx].cpu().detach().numpy(), 
            #                       y=y[idx].cpu().detach().numpy(), 
            #                       y_pred=y_pred[idx].cpu().detach().numpy())
            # for i in range(len(max_loss_idx)):
            #     idx = max_loss_idx[i]
            #     plot_trajectories(save_path=f'{save_folder}/max_{i}_{idx}.png',
            #                       x=X[0][idx].cpu().detach().numpy(), 
            #                       y=y[idx].cpu().detach().numpy(), 
            #                       y_pred=y_pred[idx].cpu().detach().numpy())
            # for i in range(len(sparse_loss_idx)):
            #     idx = sparse_loss_idx[i]
            #     plot_trajectories(save_path=f'{save_folder}/sparse_{i}_{idx}.png',
            #                       x=X[0][idx].cpu().detach().numpy(), 
            #                       y=y[idx].cpu().detach().numpy(), 
            #                       y_pred=y_pred[idx].cpu().detach().numpy())
            # exit(0)
            # # endregion

            # # region: part 2
            # bests, worsts = get_bests_worsts_vuxi(dataset_name, model_name, INIT_WINDOW, M_WINDOW, H_WINDOW, END_WINDOW)
            # if model_name == 'TRACK' or model_name == 'pos_only':
            #     bests_apd, worsts_apd = get_bests_worsts_vuxi(dataset_name, ['TRACK', 'pos_only'], INIT_WINDOW, M_WINDOW, H_WINDOW, END_WINDOW)
            #     bests += bests_apd
            #     worsts += worsts_apd
            # if model_name == 'perceiver6' or model_name == 'pos_only':
            #     bests_apd, worsts_apd = get_bests_worsts_vuxi(dataset_name, ['perceiver6', 'pos_only'], INIT_WINDOW, M_WINDOW, H_WINDOW, END_WINDOW)
            #     bests += bests_apd
            #     worsts += worsts_apd
            # save_folder = f'./plot_trajectories_figures/{dataset_name}/{model_name}/init_{INIT_WINDOW}_in_{M_WINDOW}_out_{H_WINDOW}_end_{END_WINDOW}'
            # os.makedirs(save_folder, exist_ok=True)
            # for b_i in range(batch_size):
            #     if [video[b_i], int(user[b_i]), x_i[b_i].cpu().item()] in bests+worsts:
            #         plot_trajectories(save_path=f'{save_folder}/video_{video[b_i]}_user_{user[b_i]}_x_i_{x_i[b_i].cpu().item()}.png',
            #                         x=X[0][b_i].cpu().detach().numpy(), 
            #                         y=y[b_i].cpu().detach().numpy(), 
            #                         y_pred=y_pred[b_i].cpu().detach().numpy())
            #         # 将预测结果保存为csv文件; 先录入x的数据: (x包含x, y, z3个属性)
            #         df = pd.DataFrame.from_records({'x': X[0][b_i].cpu().detach().numpy()[:, 0],
            #                                         'y': X[0][b_i].cpu().detach().numpy()[:, 1],
            #                                         'z': X[0][b_i].cpu().detach().numpy()[:, 2]})
            #         # 再继续录入y的数据: (y包含x, y, z3个属性) (不使用append方法, 因为AttributeError: 'Series' object has no attribute 'append')
            #         df = pd.concat([df, pd.DataFrame.from_records({'x': y[b_i].cpu().detach().numpy()[:, 0],
            #                                                         'y': y[b_i].cpu().detach().numpy()[:, 1],
            #                                                         'z': y[b_i].cpu().detach().numpy()[:, 2]})], ignore_index=True)
            #         # 最后继续录入y_pred的数据: (y_pred包含x, y, z3个属性) (不使用append方法, 因为AttributeError: 'Series' object has no attribute 'append')
            #         df = pd.concat([df, pd.DataFrame.from_records({'x': y_pred[b_i].cpu().detach().numpy()[:, 0],
            #                                                         'y': y_pred[b_i].cpu().detach().numpy()[:, 1],
            #                                                         'z': y_pred[b_i].cpu().detach().numpy()[:, 2]})], ignore_index=True)
                    
            #         # 计算theta和phi:
            #         df['theta'], df['phi'] = cartesian_to_eulerian(df['x'], df['y'], df['z'])

            #         df.to_csv(f'{save_folder}/video_{video[b_i]}_user_{user[b_i]}_x_i_{x_i[b_i].cpu().item()}.csv', index=False)
            # # endregion

            orth_dist = get_orth_dist(y, y_pred)  # (batch_size, h_window, 1)
            manh_dist = get_manh_dist(y, y_pred)  # manhattan distance
            acc, prec, reca, f1_score = get_acc_prec_reca_f1(y, y_pred)  # (batch_size, h_window, 1)

            for i in range(batch_size):
                for t in range(H_WINDOW):
                    errors.append({'video': video[i], 'user': user[i], 'x_i': x_i[i].cpu().item(), 't': t, 
                                   'orthodromic_distance': orth_dist[i][t][0].cpu().item(),
                                   'manhattan_distance': manh_dist[i][t][0].cpu().item(),
                                   'accuracy': acc[i][t][0].cpu().item(),
                                   'precision': prec[i][t][0].cpu().item(),
                                   'recall': reca[i][t][0].cpu().item(),
                                   'f1_score': f1_score[i][t][0].cpu().item(),
                                   })

        pickle.dump(errors, open(output_dir + f'/errors.pkl', 'wb'))
        print(f'Evaluation Finished.')
        print(f"{model_name}'s Avg. Running Time: {1000*running_time / len(test_dataloader)} ms")

        df = pd.DataFrame.from_records(errors)
        df.to_csv(output_dir + f'/errors.csv', index=False)
        # ADE_str = ''
        # for t in range(5):
        #     ADE_str += f"ADE {t + 1}s: {df[df['t'] <= t + 1]['orthodromic_distance'].mean():.3f} - "
        # print(ADE_str[:-3])

        print(f'Results saved in {output_dir}.')

