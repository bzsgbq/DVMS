import torch.nn as nn
import torch
import torch.nn.functional as F

try:
    from utils.utils import get_xyz_grid, from_position_to_tile_probability_cartesian
    from models.models_utils import get_model_size
except:
    import os
    import sys
    sys.path.append(os.path.abspath('.'))
    sys.path.append(os.path.abspath('..'))
    from utils.utils import get_xyz_grid, from_position_to_tile_probability_cartesian
    from models.models_utils import get_model_size

from models.convlstm import ConvLSTM_model


class ConvLSTMSingleModel(ConvLSTM_model):
    def __init__(self, M_WINDOW, H_WINDOW, tilemap_shape, salmap_shape, input_dim=2, hidden_dim=256, kernel_size=(3, 3), num_layers=1,
                 batch_first=True, bias=True, return_all_layers=False, block_nums=None):
        super(ConvLSTMSingleModel, self).__init__(M_WINDOW, H_WINDOW, tilemap_shape, salmap_shape, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first, bias, return_all_layers, block_nums)
        self.encoder = self.decoder


    def forward(self, x):
        """
            output_list is the stack of the hidden layer state at all points in time at the last layer
            output is [hidden layer status, cell status] at the last point in time of output_list: 
        """
        enc_pos, enc_sal, dec_pos, dec_sal = x
        batch_size = enc_pos.shape[0]
        xyz_grid = get_xyz_grid(*self.tilemap_shape)
        enc_fixmaps = from_position_to_tile_probability_cartesian(enc_pos, xyz_grid)
        enc_salmaps = self.pool(enc_sal.view(batch_size*self.m_window, 1, *self.salmap_shape)).view(batch_size, self.m_window, 1, *self.tilemap_shape)
        enc_fusemaps = torch.cat((enc_fixmaps, enc_salmaps), dim=-3)  # 将 fixmap 和 salmap 连接成一张map的两个channels

        # Encoding
        _, last_states = self.encoder(enc_fusemaps)

        # Decoding
        outputs = []
        dec_fixmap = from_position_to_tile_probability_cartesian(dec_pos, xyz_grid)

        for curr_idx in range(self.h_window):
            dec_salmap = dec_sal[:, curr_idx:curr_idx+1]
            dec_salmap = self.pool(dec_salmap.view(batch_size, 1, *self.salmap_shape)).view(batch_size, 1, 1, *self.tilemap_shape)
            dec_fusemap = torch.cat((dec_fixmap, dec_salmap), dim=-3)  # 将 fixmap 和 salmap 连接成一张map的两个channels
            out_lst, last_states = self.decoder(dec_fusemap, last_states)
            out_dec = self.outputCNN(out_lst[0])
            outputs.append(out_dec)

            # Reinject the outputs as inputs for the next loop iteration as well as update the states
            dec_fixmap = out_dec

        # Concatenate all predictions
        dec_out = torch.cat(outputs, dim=1)

        return dec_out



if __name__ == '__main__':
    B = 2
    M_WINDOW = 5
    H_WINDOW = 25
    C = 1
    H = 64
    W = 128

    model = ConvLSTMSingleModel(M_WINDOW=M_WINDOW, H_WINDOW=H_WINDOW, tilemap_shape=(9, 16), salmap_shape=(H,W), input_dim=2, hidden_dim=32, kernel_size=(3,3), num_layers=1, batch_first=True)
    enc_pos = torch.randn(B, M_WINDOW, 3)
    enc_sal = torch.randn(B, M_WINDOW, 1, H, W)
    dec_pos = torch.randn(B, 1, 3)
    dec_sal = torch.randn(B, H_WINDOW, 1, H, W)

    y = model([enc_pos, enc_sal, dec_pos, dec_sal])
    print(y.shape)

    get_model_size(model)

    # output_list, output, conv_output, conv_output_list_ret = y
    # print(output.shape)
    # print(conv_output.shape)

    # model = ConvLSTM(input_dim=1, hidden_dim=256, kernel_size=(3,3), num_layers=1, batch_first=True)
    # x = torch.randn(B, T, C, H, W)
    # output_list, output = model(x)
    # print(output_list[0].shape)

    # model = ConvLSTMCell(input_dim=1, hidden_dim=256, kernel_size=(3,3), bias=True)
    # x = torch.randn(B, C, H, W)
    # output = model(x)
    # print(output[0].shape)
