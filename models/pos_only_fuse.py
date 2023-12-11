import torch
import torch.nn as nn
import torch.optim as optim

try:
    from models.models_utils import LambdaLayer, toPosition, get_model_size
    from models.fuse import FusePosSal
    from models.pos_only import PosOnlyModel
except:
    from models_utils import LambdaLayer, toPosition, get_model_size
    from fuse import FusePosSal
    from pos_only import PosOnlyModel


class PosOnlyFuseModel(nn.Module):
    def __init__(self, M_WINDOW, H_WINDOW, tilemap_shape, salmap_shape, hidden_size=256, num_layers=1):
        super(PosOnlyFuseModel, self).__init__()
        self.m_window = M_WINDOW
        self.h_window = H_WINDOW
        self.tilemap_shape = tilemap_shape
        self.salmap_shape = salmap_shape
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.pos_model = PosOnlyModel(M_WINDOW=self.m_window, H_WINDOW=self.h_window, hidden_size=self.hidden_size, num_layers=self.num_layers)
        self.fuse_model = FusePosSal(H_WINDOW=self.h_window, tilemap_shape=self.tilemap_shape, salmap_shape=self.salmap_shape)

    def forward(self, x):
        enc_pos_in, dec_pos_in, h_sal = x
        output = self.pos_model([enc_pos_in, dec_pos_in])
        output = self.fuse_model([output, h_sal])
        return output


if __name__ == '__main__':
    batch_size = 32
    m_window = 5
    h_window = 25
    tilemap_shape = (9, 16)
    salmap_shape = (64, 128)

    model = PosOnlyFuseModel(M_WINDOW=m_window, H_WINDOW=h_window, tilemap_shape=tilemap_shape, salmap_shape=salmap_shape)
    enc_pos_input = torch.randn(batch_size, m_window, 3)
    dec_pos_input = torch.randn(batch_size, 1, 3)
    h_sal = torch.randn(batch_size, h_window, 1, *salmap_shape)
    y_pred = model([enc_pos_input, dec_pos_input, h_sal])
    print(y_pred.shape)

    get_model_size(model)