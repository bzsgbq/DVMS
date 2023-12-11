import torch
import torch.nn as nn

try:
    from Informer.model import create_informer_model
    from Informer.configs import configs as cfg
except:
    from models.Informer.model import create_informer_model
    from models.Informer.configs import configs as cfg


class InformerPlusModel(nn.Module):
    def __init__(self, M_WINDOW, H_WINDOW, salmap_shape, hidden_size=32, num_layers=2):
        super(InformerPlusModel, self).__init__()
        self.m_window = M_WINDOW
        self.h_window = H_WINDOW
        self.salmap_shape = salmap_shape
        self.salmap_height = salmap_shape[0]
        self.salmap_width = salmap_shape[1]
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # define pos informer module
        cfg.seq_len = self.m_window
        cfg.label_len = self.m_window
        cfg.pred_len = self.h_window
        cfg.enc_in = 3
        cfg.dec_in = 3
        cfg.c_out = self.hidden_size
        cfg.d_model = 128
        cfg.d_ff = 256
        cfg.dropout = 0.5
        cfg.e_layers = self.num_layers
        cfg.d_layers = self.num_layers
        cfg.s_layers = self.num_layers
        self.pos_informer = create_informer_model(cfg)

        # define sal informer module
        cfg.seq_len = self.h_window
        cfg.label_len = self.h_window
        cfg.pred_len = self.h_window
        cfg.enc_in = self.salmap_height*self.salmap_width
        cfg.dec_in = self.salmap_height*self.salmap_width
        cfg.c_out = self.hidden_size
        cfg.d_model = 64
        cfg.d_ff = 256
        cfg.dropout = 0.5
        cfg.e_layers = self.num_layers
        cfg.d_layers = self.num_layers
        cfg.s_layers = self.num_layers
        self.sal_informer = create_informer_model(cfg)

        # define fuse informer module
        cfg.seq_len = self.h_window
        cfg.label_len = self.h_window
        cfg.pred_len = self.h_window
        cfg.enc_in = self.hidden_size * 2
        cfg.dec_in = self.hidden_size * 2
        cfg.c_out = 3
        cfg.d_model = 128
        cfg.d_ff = 256
        cfg.dropout = 0.5
        cfg.e_layers = self.num_layers
        cfg.d_layers = self.num_layers
        cfg.s_layers = self.num_layers
        self.fuse_informer = create_informer_model(cfg)

        self.device = self.fuse_informer.device
    
    def forward(self, x):
        pos, sal = x

        # pos informer
        out_pos = self.pos_informer(pos)

        # sal informer
        out_sal = self.sal_informer(sal.flatten(start_dim=2))

        # fuse informer
        out = torch.cat([out_pos, out_sal], dim=-1)
        out = self.fuse_informer(out)

        return out
    

if __name__ == "__main__":
    B = 128
    m_window = 5
    h_window = 25
    salmap_shape = (64, 128)

    model = InformerPlusModel(M_WINDOW=m_window, H_WINDOW=h_window, salmap_shape=salmap_shape)
    model = model.to(model.device)
    pos = torch.randn(B, m_window, 3).to(model.device)
    sal = torch.randn(B, h_window, 1, *salmap_shape).to(model.device)
    out = model((pos, sal))
    print(out.shape)