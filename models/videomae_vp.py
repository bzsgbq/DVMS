import torch
import torch.nn as nn

import numpy as np
import torch
import torch.utils.checkpoint

from transformers.utils import (
    add_start_docstrings,
    logging,
)


logger = logging.get_logger(__name__)


VIDEOMAE_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "MCG-NJU/videomae-base",
    # See all VideoMAE models at https://huggingface.co/models?filter=videomae
]

VIDEOMAE_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`VideoMAEConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

VIDEOMAE_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`VideoMAEImageProcessor.__call__`] for details.

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


from transformers.models.videomae.modeling_videomae import VideoMAEPreTrainedModel, VideoMAEModel
from transformers import VideoMAEImageProcessor
try:
    from models_utils import get_fov, get_new_xyz
except ImportError:
    from models.models_utils import get_fov, get_new_xyz


@add_start_docstrings(
    """VideoMAE Model transformer with a Viewpoint coordinate conversion head on top (a linear layer on top of the average pooled hidden
    states of all tokens) e.g. for ImageNet.""",
    VIDEOMAE_START_DOCSTRING,
)
class VideoMAEForViewpointPrediction(VideoMAEPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.M_WINDOW = 16  # 固定为16
        self.H_WINDOW = 25  # 固定为25
        self.fov_deg = (90, 90)
        self.fov_shape = (224, 224)
        # self.image_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")

        # VideoMAE backbone
        self.videomae = VideoMAEModel(config)

        # Classifier head
        self.fc_norm = nn.LayerNorm(config.hidden_size) if config.use_mean_pooling else None
        self.classifier = nn.Linear(config.hidden_size, self.H_WINDOW*2)  # 表示2D图像上横轴和纵轴的两个坐标值

        # Initialize weights and apply final processing
        self.post_init()

        # # 冻结骨干大模型的参数
        # for param in self.videomae.parameters():
        #     param.requires_grad = False


    def forward(self, x):
        _, fov_in = x  # enc_pos_in: (batch_size, 16, 3); fov: (batch_size, 16, 3, 224, 224)
        batch_size = fov_in.shape[0]

        sequence_output = self.videomae(pixel_values=fov_in)[0]
        if self.fc_norm is not None:
            sequence_output = self.fc_norm(sequence_output.mean(1))
        else:
            sequence_output = sequence_output[:, 0]
        logits = self.classifier(sequence_output)
        vit_out = torch.tanh(logits).view(batch_size, self.H_WINDOW, 2)
        return vit_out


    def forward_backup(self, x):
        enc_pos_in, frames = x  # enc_pos_in: (batch_size, 16, 3); frames: (batch_size, 16, 512, 1024, 3)
        batch_size = enc_pos_in.shape[0]
        device = enc_pos_in.device
        h_in, w_in = frames.shape[2], frames.shape[3]
        
        all_pos = torch.zeros((batch_size, self.M_WINDOW+self.H_WINDOW, 3), dtype=torch.float32, device=device)  # (batch_size, M_WINDOW+H_WINDOW, 3)
        all_pos[:, :self.M_WINDOW, :] = enc_pos_in[:, :, :3]  # (batch_size, M_WINDOW, 3)
        del enc_pos_in
        
        fov_in = torch.zeros((batch_size, self.M_WINDOW, 3, *self.fov_shape), dtype=torch.float32, device=device)  # (batch_size, M_WINDOW, 3, 224, 224)
        _pos = all_pos[:, :self.M_WINDOW, :].reshape(-1, 3)
        _img = frames[:, :self.M_WINDOW, :, :, :].view(-1, h_in, w_in, 3)
        fov_in = get_fov(pos=_pos, img=_img, fov_deg=self.fov_deg, fov_shape=self.fov_shape)#.view(batch_size, self.M_WINDOW, *self.fov_shape, 3).permute(0, 1, 4, 2, 3)
        fov_in = fov_in.detach().cpu().numpy().astype(np.uint8)
        fov_in = self.image_processor(list(fov_in), return_tensors="pt")["pixel_values"].view(batch_size, self.M_WINDOW, 3, *self.fov_shape).to(device)  # (batch_size, M_WINDOW, 3, 224, 224)

        # 1. 一次预测所有时间步;
        sequence_output = self.videomae(pixel_values=fov_in)[0]
        if self.fc_norm is not None:
            sequence_output = self.fc_norm(sequence_output.mean(1))
        else:
            sequence_output = sequence_output[:, 0]
        logits = self.classifier(sequence_output)
        vit_out = torch.tanh(logits).view(batch_size, self.H_WINDOW, 2)
        return vit_out
        # for t in range(self.H_WINDOW):
        #     old_xyz = all_pos[:, t+self.M_WINDOW-1].permute(1, 0)  # xyz是当前已知的最新视点的三维空间坐标
        #     new_xyz = get_new_xyz(old_xyz, vit_out[:, t], self.fov_deg)
        #     all_pos[:, t+self.M_WINDOW] = new_xyz.permute(1, 0)
        # return all_pos[:, self.M_WINDOW:, :]

        # # 2. 递归预测所有时间步;
        # for t in range(self.H_WINDOW):
        #     sequence_output = self.videomae(pixel_values=fov_in)[0]
        #     if self.fc_norm is not None:
        #         sequence_output = self.fc_norm(sequence_output.mean(1))
        #     else:
        #         sequence_output = sequence_output[:, 0]
        #     logits = self.classifier(sequence_output)
        #     vit_out = torch.tanh(logits)

        #     old_xyz = all_pos[:, t+self.M_WINDOW-1].permute(1, 0)  # xyz是当前已知的最新视点的三维空间坐标
        #     new_xyz = get_new_xyz(old_xyz, vit_out, self.fov_deg)
        #     all_pos[:, t+self.M_WINDOW] = new_xyz.permute(1, 0)
        #     new_fov = get_fov(all_pos[:, t+self.M_WINDOW], frames[:, t+self.M_WINDOW], fov_deg=self.fov_deg, fov_shape=self.fov_shape).permute(0, 3, 1, 2)
        #     fov_in = torch.cat([fov_in[:, 1:], new_fov.unsqueeze(1)], dim=1)

        # return all_pos[:, self.M_WINDOW:, :]


    def predict(self, x):
        enc_pos_in, _ = x  # enc_pos_in: (batch_size, 16, 3); frames: (batch_size, 16, 512, 1024, 3); fov: (batch_size, 16, 3, 224, 224)
        device = enc_pos_in.device
        y_pred = self.forward(x)
        
        y_temp = torch.zeros((y_pred.shape[0], y_pred.shape[1], 3), dtype=torch.float32, device=device)
        old_xyz = enc_pos_in[:, -1, :3].permute(1, 0)
        for t in range(self.H_WINDOW):
            y_temp[:, t, :] = get_new_xyz(old_xyz, y_pred[:, t, :], fov_deg=self.fov_deg).permute(1, 0)
            old_xyz = y_temp[:, t, :].permute(1, 0)
        y_pred = y_temp
        
        return y_pred



if __name__ == '__main__':
    batch_size = 2
    device = 'cuda:0'

    model_ckpt = "MCG-NJU/videomae-base-finetuned-kinetics"
    model = VideoMAEForViewpointPrediction.from_pretrained(model_ckpt, ignore_mismatched_sizes=True).to(device)

    enc_pos_in = torch.tensor([1, 0, 0], dtype=torch.float32).to(device).repeat(batch_size, 16, 1)
    frames = torch.rand((batch_size, 16, 512, 1024, 3)).to(device) * 255
    fov = torch.rand((batch_size, 16, 3, 224, 224)).to(device)

    # memory_usage = frames.element_size() * frames.nelement()  # element_size() 返回每个元素的字节大小，nelement() 返回元素数量
    # print("frames显存占用大小:", memory_usage / 1024 / 1024, "MB")
    # exit()

    # # 开始模型的前向传播，以测量显存占用
    # torch.cuda.empty_cache()  # 清除缓存
    # model.eval()  # 设置模型为评估模式，以避免使用额外的显存（如梯度等）
    # with torch.no_grad():  # 禁用梯度计算
    #     output = model((enc_pos_in, frames))
    #     memory_usage = torch.cuda.memory_allocated('cuda:0')  # 获取当前已分配的显存大小

    # print("模型实际需要的显存空间:", memory_usage / 1024 / 1024, "MB")
    # exit()

    # model.eval()  # 设置模型为评估模式，以避免使用额外的显存（如梯度等）
    with torch.no_grad():  # 禁用梯度计算
        # # 第一种:
        # y = model((enc_pos_in, frames))
        # print(y.shape)
        # y_pred = model.predict((enc_pos_in, frames))
        # print(y_pred.shape)

        # 第二种:
        y = model((enc_pos_in, fov))
        print(y.shape)
        y_pred = model.predict((enc_pos_in, fov))
        print(y_pred.shape)
    