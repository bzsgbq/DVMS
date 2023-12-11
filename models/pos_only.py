import torch
import torch.nn as nn
import torch.optim as optim

try:
    from models.models_utils import LambdaLayer, toPosition, get_model_size, get_new_xyz
except:
    from models_utils import LambdaLayer, toPosition, get_model_size, get_new_xyz


num_feat_dim = 3
assert num_feat_dim in [2, 3]


class PosOnlyModel(nn.Module):
    def __init__(self, M_WINDOW, H_WINDOW, hidden_size=256, num_layers=2):
        super(PosOnlyModel, self).__init__()
        self.m_window = M_WINDOW
        self.h_window = H_WINDOW
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.sense_pos_enc = nn.LSTM(input_size=num_feat_dim, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.sense_pos_dec = nn.LSTM(input_size=num_feat_dim, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc_layer_out = nn.Linear(self.hidden_size, num_feat_dim)

        self.to_position = LambdaLayer(toPosition)

    def forward(self, x):
        encoder_position_inputs, decoder_position_inputs = x
        encoder_position_inputs = encoder_position_inputs[:, :, -2:] if num_feat_dim == 2 else encoder_position_inputs[:, :, :3]
        decoder_position_inputs = decoder_position_inputs[:, :, -2:] if num_feat_dim == 2 else decoder_position_inputs[:, :, :3]

        # Encoding
        _, state_1 = self.sense_pos_enc(encoder_position_inputs)

        # Decoding
        all_pos_outputs = []
        inputs = decoder_position_inputs
        for curr_idx in range(self.h_window):
            out_dec_pos, state_1 = self.sense_pos_dec(inputs, state_1)
            outputs_delta = self.fc_layer_out(out_dec_pos)
            decoder_pred = outputs_delta if num_feat_dim == 2 else self.to_position([inputs, outputs_delta])

            all_pos_outputs.append(decoder_pred)

            # Reinject the outputs as inputs for the next loop iteration as well as update the states
            inputs = decoder_pred

        # Concatenate all predictions
        decoder_outputs_pos = torch.cat(all_pos_outputs, dim=1)
        
        if num_feat_dim == 2:
            decoder_outputs_pos = torch.tanh(decoder_outputs_pos)

        return decoder_outputs_pos
    

    def predict(self, x):
        encoder_position_inputs, decoder_position_inputs = x
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


if __name__ == '__main__':
    model = PosOnlyModel(M_WINDOW=5, H_WINDOW=25)
    enc_pos_input = torch.randn(32, 5, 5 if num_feat_dim == 2 else 3)
    dec_pos_input = torch.randn(32, 1, 5 if num_feat_dim == 2 else 3)
    y_pred = model([enc_pos_input, dec_pos_input])
    print(y_pred.shape)
    y_pred = model.predict([enc_pos_input, dec_pos_input])
    print(y_pred.shape)
    get_model_size(model)