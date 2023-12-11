import torch
import torch.nn as nn

try:
    from models.models_utils import LambdaLayer, toPosition, get_model_size, get_fov, get_new_xyz
    from models.sal_convs import SalConvs
except:
    from models_utils import LambdaLayer, toPosition, get_model_size, get_fov, get_new_xyz
    from sal_convs import SalConvs


USE_CNN = True
USE_FUSE_FC = True

num_feat_dim = 2
assert num_feat_dim in [2]


class TRACKMaskModel(nn.Module):
    def __init__(self, M_WINDOW, H_WINDOW, salmap_shape, hidden_size=256, num_layers=1):
        super(TRACKMaskModel, self).__init__()
        self.m_window = M_WINDOW
        self.h_window = H_WINDOW
        self.salmap_shape = salmap_shape
        self.fov_deg = (90, 90)
        self.fov_shape = (64, 64)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.sense_pos_enc = nn.LSTM(input_size=num_feat_dim, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.sense_sal_enc = nn.LSTM(input_size=self.hidden_size if USE_CNN else self.fov_shape[0]*self.fov_shape[1], hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fuse_1_enc = nn.LSTM(input_size=self.hidden_size*2, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)

        self.sense_pos_dec = nn.LSTM(input_size=num_feat_dim, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.sense_sal_dec = nn.LSTM(input_size=self.hidden_size if USE_CNN else self.fov_shape[0]*self.fov_shape[1], hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fuse_1_dec = nn.LSTM(input_size=self.hidden_size*2, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)

        self.fuse_2 = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)

        self.fc_layer_out = nn.Linear(in_features=self.hidden_size, out_features=num_feat_dim)
        
        if USE_FUSE_FC:
            self.fc_fuse_enc = nn.Sequential(
                nn.Linear(in_features=2*self.hidden_size, out_features=2*self.hidden_size),
                nn.ReLU(),
                nn.Linear(in_features=2*self.hidden_size, out_features=2*self.hidden_size)
            )
            self.fc_fuse_dec = nn.Sequential(
                nn.Linear(in_features=2*self.hidden_size, out_features=2*self.hidden_size),
                nn.ReLU(),
                nn.Linear(in_features=2*self.hidden_size, out_features=2*self.hidden_size)
            )

        if USE_CNN:
            self.sal_convs_enc = SalConvs(salmap_shape=self.fov_shape, init_ch_size=16, output_size=self.hidden_size, num_conv_layer=3, padding_mode='zeros')
            self.sal_convs_dec = SalConvs(salmap_shape=self.fov_shape, init_ch_size=16, output_size=self.hidden_size, num_conv_layer=3, padding_mode='zeros')
        
        self.to_position = LambdaLayer(toPosition)

    
    def forward(self, x):
        encoder_position_inputs, encoder_saliency_inputs, decoder_position_inputs, decoder_saliency_inputs = x

        # Encoding
        out_enc_pos, states_1 = self.sense_pos_enc(encoder_position_inputs[:, :, -num_feat_dim:])
        
        _pos = encoder_position_inputs.view(-1, 5)[:, :3]
        _img = encoder_saliency_inputs.view(-1, 1, *self.salmap_shape).permute(0, 2, 3, 1)  # (B*T, H, W, C)
        encoder_saliency_inputs = get_fov(_pos, _img, self.fov_deg, self.fov_shape).view(-1, self.m_window, *self.fov_shape, 1).permute(0, 1, 4, 2, 3)  # (B, T, C, H, W)
        if not USE_CNN:
            out_flat_enc = encoder_saliency_inputs.flatten(start_dim=2)
        else:
            out_flat_enc = self.sal_convs_enc(encoder_saliency_inputs)
        out_enc_sal, states_2 = self.sense_sal_enc(out_flat_enc)

        conc_out_enc = torch.cat([out_enc_sal, out_enc_pos], dim=-1)
        if USE_FUSE_FC:
            conc_out_enc = self.fc_fuse_enc(conc_out_enc)
        fuse_out_enc, states_fuse = self.fuse_1_enc(conc_out_enc)

        # Decoding
        all_pos_outputs = []
        fovxy_pred = decoder_position_inputs[:, :, -2:]  # (B, 1, 2)
        old_xyz = decoder_position_inputs[:, 0, :3].permute(1, 0)  # (3, B)
        for curr_idx in range(self.h_window):
            out_enc_pos, states_1 = self.sense_pos_dec(fovxy_pred, states_1)

            selected_timestep_saliency = decoder_saliency_inputs[:, curr_idx:curr_idx+1]
            _pos = old_xyz.permute(1, 0)  # (B, 3)
            _img = selected_timestep_saliency.view(-1, 1, *self.salmap_shape).permute(0, 2, 3, 1)  # (B, H, W, C)
            selected_timestep_saliency = get_fov(_pos, _img, self.fov_deg, self.fov_shape).view(-1, 1, *self.fov_shape, 1).permute(0, 1, 4, 2, 3)  # (B, T, C, H, W)
            if not USE_CNN:
                flatten_timestep_saliency = selected_timestep_saliency.view(selected_timestep_saliency.shape[0], 1, -1)
            else:
                flatten_timestep_saliency = self.sal_convs_dec(selected_timestep_saliency)
            out_enc_sal, states_2 = self.sense_sal_dec(flatten_timestep_saliency, states_2)

            conc_out_dec = torch.cat([out_enc_sal, out_enc_pos], dim=-1)
            if USE_FUSE_FC:
                conc_out_dec = self.fc_fuse_dec(conc_out_dec)
            fuse_out_dec_1, states_fuse = self.fuse_1_dec(conc_out_dec, states_fuse)
            fuse_out_dec_2 = self.fuse_2(fuse_out_dec_1)

            fovxy_pred = self.fc_layer_out(fuse_out_dec_2)
            old_xyz = get_new_xyz(old_xyz, fovxy_pred.squeeze(1), self.fov_deg)
            all_pos_outputs.append(fovxy_pred)

        # Concatenate all predictions
        decoder_outputs_pos = torch.cat(all_pos_outputs, dim=1)

        return decoder_outputs_pos


    def predict(self, x):
        encoder_position_inputs, encoder_saliency_inputs, decoder_position_inputs, decoder_saliency_inputs = x
        device = encoder_position_inputs.device
        y_pred = self.forward(x)
        if num_feat_dim == 2:
            y_temp = torch.zeros((y_pred.shape[0], y_pred.shape[1], 3), dtype=torch.float32, device=device)
            old_xyz = decoder_position_inputs[:, -1, :3].permute(1, 0)
            for t in range(self.h_window):
                y_temp[:, t, :] = get_new_xyz(old_xyz, y_pred[:, t, :], fov_deg=(90, 90)).permute(1, 0)
                old_xyz = y_temp[:, t, :].permute(1, 0)
            y_pred = y_temp
        return y_pred



if __name__ == "__main__":
    B = 32
    M_WINDOW, H_WINDOW = 5, 25
    SALMAP_SHAPE = (256, 512)

    encoder_position_inputs = torch.randn(B, M_WINDOW, 5 if num_feat_dim == 2 else 3)
    encoder_saliency_inputs = torch.randn(B, M_WINDOW, 1, *SALMAP_SHAPE)
    decoder_position_inputs = torch.randn(B, 1, 5 if num_feat_dim == 2 else 3)
    decoder_saliency_inputs = torch.randn(B, H_WINDOW, 1, *SALMAP_SHAPE)
    
    model = TRACKMaskModel(M_WINDOW, H_WINDOW, SALMAP_SHAPE)
    
    decoder_outputs_pos = model([encoder_position_inputs, encoder_saliency_inputs, decoder_position_inputs, decoder_saliency_inputs])
    print(decoder_outputs_pos.shape)

    y_pred = model.predict([encoder_position_inputs, encoder_saliency_inputs, decoder_position_inputs, decoder_saliency_inputs])
    print(y_pred.shape)

    get_model_size(model)