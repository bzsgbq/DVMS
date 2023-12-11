import torch
import torch.nn as nn
import torch.optim as optim

try:
    from models.models_utils import LambdaLayer, toPosition, get_model_size
except:
    from models_utils import LambdaLayer, toPosition, get_model_size


class PosOnlySingleModel(nn.Module):
    def __init__(self, M_WINDOW, H_WINDOW, hidden_size=256, num_layers=1):
        super(PosOnlySingleModel, self).__init__()
        self.m_window = M_WINDOW
        self.h_window = H_WINDOW
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.sense_pos = nn.LSTM(input_size=3, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc_layer_out = nn.Linear(self.hidden_size, 3)

        self.to_position = LambdaLayer(toPosition)

    def forward(self, x):
        encoder_position_inputs, decoder_position_inputs = x

        # Encoding
        _, state_1 = self.sense_pos(encoder_position_inputs)

        # Decoding
        all_pos_outputs = []
        inputs = decoder_position_inputs
        for curr_idx in range(self.h_window):
            out_dec_pos, state_1 = self.sense_pos(inputs, state_1)
            outputs_delta = self.fc_layer_out(out_dec_pos)
            decoder_pred = self.to_position([inputs, outputs_delta])

            all_pos_outputs.append(decoder_pred)

            # Reinject the outputs as inputs for the next loop iteration as well as update the states
            inputs = decoder_pred

        # Concatenate all predictions
        decoder_outputs_pos = torch.cat(all_pos_outputs, dim=1)

        return decoder_outputs_pos


if __name__ == '__main__':
    model = PosOnlySingleModel(M_WINDOW=5, H_WINDOW=25)
    enc_pos_input = torch.randn(32, 5, 3)
    dec_pos_input = torch.randn(32, 1, 3)
    y_pred = model([enc_pos_input, dec_pos_input])
    print(y_pred.shape)

    get_model_size(model)