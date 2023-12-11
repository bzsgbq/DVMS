import torch
import torch.nn as nn

try:
    from models.models_utils import LambdaLayer, toPosition, get_model_size, get_new_xyz
except:
    from models_utils import LambdaLayer, toPosition, get_model_size, get_new_xyz


num_feat_dim = 3
assert num_feat_dim in [2, 3]


class MM18Model(nn.Module):
    def __init__(self, M_WINDOW, H_WINDOW, salmap_shape, hidden_size=256, num_layers=2):
        super(MM18Model, self).__init__()
        self.m_window = M_WINDOW
        self.h_window = H_WINDOW
        self.salmap_height = salmap_shape[0]
        self.salmap_width = salmap_shape[1]
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # self.sense_pos_enc = nn.LSTM(input_size=num_feat_dim, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc_pos_enc = nn.Linear(in_features=num_feat_dim, out_features=self.hidden_size)
        # self.sense_sal_enc = nn.LSTM(input_size=self.salmap_height*self.salmap_width, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc_sal_enc = nn.Linear(in_features=self.salmap_height*self.salmap_width, out_features=self.hidden_size)
        self.fuse_1_enc = nn.LSTM(input_size=2*self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        
        # self.sense_pos_dec = nn.LSTM(input_size=num_feat_dim, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc_pos_dec = nn.Linear(in_features=num_feat_dim, out_features=self.hidden_size)
        # self.sense_sal_dec = nn.LSTM(input_size=self.salmap_height*self.salmap_width, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc_sal_dec = nn.Linear(in_features=self.salmap_height*self.salmap_width, out_features=self.hidden_size)
        self.fuse_1_dec = nn.LSTM(input_size=2*self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fuse_2 = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
        self.fc_layer_out = nn.Linear(in_features=self.hidden_size, out_features=num_feat_dim)
        
        self.to_position = LambdaLayer(toPosition)


    def forward(self, x):
        encoder_position_inputs, encoder_saliency_inputs, decoder_position_inputs, decoder_saliency_inputs = x
        encoder_position_inputs = encoder_position_inputs[:, :, -2:] if num_feat_dim == 2 else encoder_position_inputs[:, :, :3]
        decoder_position_inputs = decoder_position_inputs[:, :, -2:] if num_feat_dim == 2 else decoder_position_inputs[:, :, :3]

        # Encoding
        # out_enc_pos, states_1 = self.sense_pos_enc(encoder_position_inputs)
        out_enc_pos = self.fc_pos_enc(encoder_position_inputs)

        out_flat_enc = encoder_saliency_inputs.flatten(start_dim=2)
        # out_enc_sal, states_2 = self.sense_sal_enc(out_flat_enc)
        out_enc_sal = self.fc_sal_enc(out_flat_enc)

        conc_out_enc = torch.cat([out_enc_sal, out_enc_pos], dim=-1)
        fuse_out_enc, states_fuse = self.fuse_1_enc(conc_out_enc)

        # Decoding
        all_pos_outputs = []
        inputs = decoder_position_inputs
        for curr_idx in range(self.h_window):
            # out_enc_pos, states_1 = self.sense_pos_dec(inputs, states_1)
            out_enc_pos = self.fc_pos_dec(inputs)

            selected_timestep_saliency = decoder_saliency_inputs[:, curr_idx:curr_idx+1]
            flatten_timestep_saliency = selected_timestep_saliency.flatten(start_dim=2)#.view(selected_timestep_saliency.shape[0], 1, -1)
            # out_enc_sal, states_2 = self.sense_sal_dec(flatten_timestep_saliency, states_2)
            out_enc_sal = self.fc_sal_dec(flatten_timestep_saliency)

            conc_out_dec = torch.cat([out_enc_sal, out_enc_pos], dim=-1)

            fuse_out_dec_1, states_fuse = self.fuse_1_dec(conc_out_dec, states_fuse)
            fuse_out_dec_2 = self.fuse_2(fuse_out_dec_1)

            outputs_delta = self.fc_layer_out(fuse_out_dec_2)

            decoder_pred = outputs_delta if num_feat_dim == 2 else self.to_position([inputs, outputs_delta])

            all_pos_outputs.append(decoder_pred)
            # Reinject the outputs as inputs for the next loop iteration as well as update the states
            inputs = decoder_pred

        # Concatenate all predictions
        decoder_outputs_pos = torch.cat(all_pos_outputs, dim=1)

        # if num_feat_dim == 2:
        #     decoder_outputs_pos = torch.tanh(decoder_outputs_pos)

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
    SALMAP_SHAPE = (64, 128)

    encoder_position_inputs = torch.randn(B, M_WINDOW, 5 if num_feat_dim == 2 else 3)
    encoder_saliency_inputs = torch.randn(B, M_WINDOW, 1, *SALMAP_SHAPE)
    decoder_position_inputs = torch.randn(B, 1, 5 if num_feat_dim == 2 else 3)
    decoder_saliency_inputs = torch.randn(B, H_WINDOW, 1, *SALMAP_SHAPE)
    
    model = MM18Model(M_WINDOW, H_WINDOW, SALMAP_SHAPE)
    
    decoder_outputs_pos = model([encoder_position_inputs, encoder_saliency_inputs, decoder_position_inputs, decoder_saliency_inputs])
    print(decoder_outputs_pos.shape)

    y_pred = model.predict([encoder_position_inputs, encoder_saliency_inputs, decoder_position_inputs, decoder_saliency_inputs])
    print(y_pred.shape)

    get_model_size(model)