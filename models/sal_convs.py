import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from models.models_utils import get_model_size
except:
    from models_utils import get_model_size


class SalConvs(nn.Module):
    def __init__(self, salmap_shape, init_ch_size, output_size, num_conv_layer=None, padding_mode='circular'):
        super(SalConvs, self).__init__()

        ch_size = init_ch_size
        if num_conv_layer is None:
            num_conv_layer = min(int(math.log(salmap_shape[0], 2)), int(math.log(salmap_shape[1], 2)))
            assert num_conv_layer >= 2
        final_map_size = int(salmap_shape[0] // math.pow(2, num_conv_layer)) * int(salmap_shape[1] // math.pow(2, num_conv_layer))
        assert final_map_size > 0 and output_size % final_map_size == 0
        self.convs = nn.ModuleList([nn.Conv2d(1, ch_size, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)])
        for _ in range(1, num_conv_layer):
            self.convs.append(nn.Conv2d(ch_size, 2*ch_size, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode))
            ch_size *= 2
        
        self.fc = nn.Linear(final_map_size * ch_size, output_size)

    def forward(self, x):
        '''
        case1:
            x.shape = (batch_size, 1, H, W)
            return.shape = (batch_size, output_size)
        case2:
            x.shape = (batch_size, seq_len, 1, H, W)
            return.shape = (batch_size, seq_len, output_size)
        '''
        if len(x.shape) == 4:
            for conv in self.convs:
                x = conv(x)
                x = nn.functional.relu(x)
                x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
            x = x.flatten(start_dim=1)
            x = self.fc(x)
            return x

        elif len(x.shape) == 5:
            h_window = x.shape[1]
            outs = []
            for t in range(h_window):
                out = x[:, t]
                out = self.forward(out)
                outs.append(out)
            outs = torch.stack(outs, dim=1) # (batch_size, h_window, output_size)
            return outs


if __name__ == '__main__':
    SALMAP_SHAPE = (64, 128)
    B = 128
    H_WINDOW = 25
    
    model = SalConvs(salmap_shape=SALMAP_SHAPE, init_ch_size=16, output_size=256, num_conv_layer=4)
    x = torch.randn(B, H_WINDOW, 1, *SALMAP_SHAPE)
    y = model(x)
    print(y.shape)

    get_model_size(model)

    # x = 100
    # while x > 1:
    #     print(x)
    #     x = int(x // 2)
    # print(1)