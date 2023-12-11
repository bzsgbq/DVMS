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


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              padding_mode='circular',
                              bias=self.bias)


    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []

        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)


    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        ----------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            # raise NotImplementedError()
            pass
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)      # Hidden layer states at all points in time
            last_state_list.append([h, c])              # Hidden layer status and cell status at the last point in time

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

class outputCNN(nn.Module):
    def __init__(self, input_dim):
        super(outputCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=128, kernel_size=(5, 5), padding=(2, 2))
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), padding=(2, 2))
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(5, 5), padding=(2, 2))
        self.conv4 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=(5, 5), padding=(2, 2))

    def forward(self,x):
        if len(x.shape) == 4:
            x = F.relu(self.conv1(x))
            output_size = x.shape
            x, i = self.pool(x)
            x = F.relu(self.conv2(x))
            x = self.unpool(x, i, output_size=output_size)
            x = F.relu(self.conv3(x))
            # x = torch.sigmoid(self.conv4(x))
            x = torch.relu(self.conv4(x))
            return x
        elif len(x.shape) == 5:
            B, T, C, H, W = x.shape
            x = x.view(B*T, C, H, W)
            x = F.relu(self.conv1(x))
            output_size = x.shape
            x, i = self.pool(x)
            x = F.relu(self.conv2(x))
            x = self.unpool(x, i, output_size=output_size)
            x = F.relu(self.conv3(x))
            # x = torch.sigmoid(self.conv4(x))
            x = torch.relu(self.conv4(x))
            x = x.view(B, T, 1, H, W)
            return x
        else:
            raise ValueError('The input tensor must be 4-D or 5-D')


class ConvLSTM_model(nn.Module):
    def __init__(self, M_WINDOW, H_WINDOW, tilemap_shape, salmap_shape, input_dim=2, hidden_dim=256, kernel_size=(3, 3), num_layers=1,
                 batch_first=True, bias=True, return_all_layers=False, block_nums=None):
        super(ConvLSTM_model, self).__init__()

        self.m_window = M_WINDOW
        self.h_window = H_WINDOW
        self.seq_len = H_WINDOW
        self.tilemap_shape = tilemap_shape
        self.salmap_shape = salmap_shape
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        ks = (salmap_shape[0]//tilemap_shape[0], salmap_shape[1]//tilemap_shape[1])
        self.pool = nn.AvgPool2d(kernel_size=ks, stride=ks)

        self.encoder = ConvLSTM(self.input_dim, self.hidden_dim, self.kernel_size, self.num_layers, 
                                self.batch_first, self.bias, self.return_all_layers)
        self.decoder = ConvLSTM(self.input_dim, self.hidden_dim, self.kernel_size, self.num_layers, 
                                self.batch_first, self.bias, self.return_all_layers)
        self.outputCNN = outputCNN(self.hidden_dim)

    def initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


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

    model = ConvLSTM_model(M_WINDOW=M_WINDOW, H_WINDOW=H_WINDOW, tilemap_shape=(9, 16), salmap_shape=(H,W), input_dim=2, hidden_dim=32, kernel_size=(3,3), num_layers=1, batch_first=True)
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
