import torch
import torch.nn as nn
import torch.optim as optim

try:
    from models.models_utils import LambdaLayer, toPosition, get_model_size, get_new_xyz
except:
    from models_utils import LambdaLayer, toPosition, get_model_size, get_new_xyz


num_feat_dim = 3
assert num_feat_dim in [2, 3]


class ReactiveModel(nn.Module):
    def __init__(self, M_WINDOW, H_WINDOW):
        super(ReactiveModel, self).__init__()
        self.m_window = M_WINDOW
        self.h_window = H_WINDOW
        # 随便设置一个不用的线性层占位:
        self.linear = nn.Linear(num_feat_dim, num_feat_dim)

    def forward(self, x):
        # 进行假训练; 仅仅是为了能够让程序正常运行, 无实际意义:
        encoder_position_inputs, _ = x
        encoder_position_inputs = encoder_position_inputs[:, :, -2:] if num_feat_dim == 2 else encoder_position_inputs[:, :, :3]
        return self.linear(encoder_position_inputs[:, -1, :].unsqueeze(1).expand(-1, self.h_window, -1))

    def predict(self, x):
        # 实际预测时, 直接返回最后一个输入的位置:
        encoder_position_inputs, _ = x
        encoder_position_inputs = encoder_position_inputs[:, :, -2:] if num_feat_dim == 2 else encoder_position_inputs[:, :, :3]
        y_pred = encoder_position_inputs[:, -1, :].unsqueeze(1).expand(-1, self.h_window, -1)
        return y_pred


if __name__ == '__main__':
    model = ReactiveModel(15, 25)
    enc_pos_input = torch.randn(32, 15, 5 if num_feat_dim == 2 else 3)
    dec_pos_input = torch.randn(32, 1, 5 if num_feat_dim == 2 else 3)
    y_pred = model([enc_pos_input, dec_pos_input])
    print(y_pred.shape)
    y_pred = model.predict([enc_pos_input, dec_pos_input])
    print(y_pred.shape)
    print(enc_pos_input[0, -1, :])
    print(y_pred[0])
    get_model_size(model)