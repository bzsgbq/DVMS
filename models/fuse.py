import torch
import torch.nn as nn
try:
    from utils.utils import get_xyz_grid, from_position_to_tile_probability_cartesian
except:
    import os
    import sys
    sys.path.append(os.path.abspath('.'))
    sys.path.append(os.path.abspath('..'))
    from utils.utils import get_xyz_grid, from_position_to_tile_probability_cartesian


class FusePosSal(nn.Module):
    def __init__(self, H_WINDOW, tilemap_shape=(9, 16), salmap_shape=(256, 256)):
        super(FusePosSal, self).__init__()
        self.h_window = H_WINDOW
        self.seq_len = H_WINDOW
        self.tilemap_shape = tilemap_shape
        self.salmap_shape = salmap_shape

        # self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1) for _ in range(self.seq_len)])
        self.relu = nn.ReLU()
        ks = (salmap_shape[0]//tilemap_shape[0], salmap_shape[1]//tilemap_shape[1])
        self.pool = nn.AvgPool2d(kernel_size=ks, stride=ks)
        # self.fc = nn.Linear(in_features=128*32*32, out_features=9*16)
    
    def forward(self, x):
        pos, sal = x
        # pos.shape = (batch_size, seq_len, 3)
        # sal.shape = (batch_size, seq_len, height, width, 1)
        batch_size = pos.shape[0]
        xyz_grid = get_xyz_grid(*self.tilemap_shape)
        fixmap = from_position_to_tile_probability_cartesian(pos, xyz_grid)
        salmap = self.pool(sal.view(batch_size*self.seq_len, 1, *self.salmap_shape)).view(batch_size, self.seq_len, 1, *self.tilemap_shape)

        fusemap = torch.cat((fixmap, salmap), dim=-3)  # 将 fixmap 和 salmap 连接成一张map的两个channels

        outputs = []
        for i in range(self.seq_len):
            x = fusemap[:, i, :, :, :]
            x = self.convs[i](x)
            x = self.relu(x)
            outputs.append(x)
        
        return torch.stack(outputs, dim=1)


if __name__ == '__main__':
    batch_size = 4
    h_window = 5
    tilemap_shape = (9, 16)
    salmap_shape = (256, 256)
    model = FusePosSal(H_WINDOW=h_window, tilemap_shape=tilemap_shape, salmap_shape=salmap_shape)
    pos = torch.randn(batch_size, h_window, 3)
    sal = torch.randn(batch_size, h_window, *salmap_shape, 1)
    y = model([pos, sal])
    print(y.shape)