import torch
import torch.nn as nn
import numpy as np


class PosOnlyModel(nn.Module):
    def __init__(self, M_WINDOW, H_WINDOW):
        super(PosOnlyModel, self).__init__()
        self.lstm_layer = nn.LSTM(input_size=2, hidden_size=1024, num_layers=1, batch_first=True)
        self.decoder_dense_mot = nn.Sequential(nn.Linear(1024, 2), nn.Sigmoid())
        self.decoder_dense_dir = nn.Sequential(nn.Linear(1024, 2), nn.Tanh())
        self.to_position = ToPosition()
        self.M_WINDOW = M_WINDOW
        self.H_WINDOW = H_WINDOW

    def forward(self, encoder_inputs, decoder_inputs):
        # Encoding
        encoder_outputs, (state_h, state_c) = self.lstm_layer(encoder_inputs)

        # Decoding
        all_outputs = []
        inputs = decoder_inputs
        states = (state_h, state_c)
        for curr_idx in range(self.H_WINDOW):
            # Run the decoder on one timestep
            decoder_pred, (state_h, state_c) = self.lstm_layer(inputs, states)
            outputs_delta = self.decoder_dense_mot(decoder_pred)
            outputs_delta_dir = self.decoder_dense_dir(decoder_pred)
            outputs_pos = self.to_position(inputs, outputs_delta, outputs_delta_dir)
            # Store the current prediction (we will concantenate all predictions later)
            all_outputs.append(outputs_pos)
            # Reinject the outputs as inputs for the next loop iteration as well as update the states
            inputs = outputs_pos
            states = (state_h, state_c)

        # Concatenate all predictions
        decoder_outputs = torch.cat(all_outputs, dim=1)

        return decoder_outputs


class ToPosition(nn.Module):
    def __init__(self):
        super(ToPosition, self).__init__()

    def forward(self, values, outputs_delta, outputs_delta_dir):
        orientation = values[:, :, 0:2]
        magnitudes = outputs_delta / 2.0
        directions = outputs_delta_dir
        # The network returns values between 0 and 1, we force it to be between -2/5 and 2/5
        motion = magnitudes * directions

        yaw_pred_wo_corr = orientation[:, :, 0:1] + motion[:, :, 0:1]
        pitch_pred_wo_corr = orientation[:, :, 1:2] + motion[:, :, 1:2]

        cond_above = (pitch_pred_wo_corr > 1.0).float()
        cond_correct = ((pitch_pred_wo_corr <= 1.0) & (pitch_pred_wo_corr >= 0.0)).float()
        cond_below = (pitch_pred_wo_corr < 0.0).float()

        pitch_pred = cond_above * (1.0 - (pitch_pred_wo_corr - 1.0)) + cond_correct * pitch_pred_wo_corr + cond_below * (-pitch_pred_wo_corr)
        yaw_pred = torch.fmod(cond_above * (yaw_pred_wo_corr - 0.5) + cond_correct * yaw_pred_wo_corr + cond_below * (yaw_pred_wo_corr - 0.5), 1.0)

        return torch.cat([yaw_pred, pitch_pred], dim=-1)


if __name__ == '__main__':
    # Test the model
    model = PosOnlyModel(5, 25)
    encoder_inputs = torch.rand(32, 5, 2)
    decoder_inputs = torch.rand(32, 1, 2)
    outputs = model(encoder_inputs, decoder_inputs)
    print(outputs.shape)
