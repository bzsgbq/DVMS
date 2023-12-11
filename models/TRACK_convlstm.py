import torch
import torch.nn as nn
import torch.optim as optim

try:
    from models.models_utils import MaxPool, LambdaLayer, toPosition, get_model_size
    from models.convlstm import ConvLSTM
except:
    from models_utils import MaxPool, LambdaLayer, toPosition, get_model_size
    from convlstm import ConvLSTM


class TRACKConvLstmModel(nn.Module):
    def __init__(self, M_WINDOW, H_WINDOW, salmap_shape, hidden_size=256, num_layers=1):
        super(TRACKConvLstmModel, self).__init__()
        self.m_window = M_WINDOW
        self.h_window = H_WINDOW
        self.salmap_height = salmap_shape[0]
        self.salmap_width = salmap_shape[1]
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.sense_pos_enc = nn.LSTM(input_size=3, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        # self.sense_sal_enc = nn.LSTM(input_size=self.salmap_height*self.salmap_width, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.sense_sal_enc = ConvLSTM(input_dim=1, hidden_dim=4, kernel_size=(3, 3), num_layers=self.num_layers, batch_first=True, bias=True, return_all_layers=False)
        self.fc_sal_enc = nn.Linear(in_features=self.salmap_height*self.salmap_width, out_features=self.hidden_size)
        self.fuse_1_enc = nn.LSTM(input_size=2*self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.sense_pos_dec = nn.LSTM(input_size=3, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        # self.sense_sal_dec = nn.LSTM(input_size=self.salmap_height*self.salmap_width, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.sense_sal_dec = ConvLSTM(input_dim=1, hidden_dim=4, kernel_size=(3, 3), num_layers=self.num_layers, batch_first=True, bias=True, return_all_layers=False)
        self.fc_sal_dec = nn.Linear(in_features=self.salmap_height*self.salmap_width, out_features=self.hidden_size)
        self.fuse_1_dec = nn.LSTM(input_size=2*self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fuse_2 = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
        self.fc_layer_out = nn.Linear(in_features=self.hidden_size, out_features=3)
        
        self.to_position = LambdaLayer(toPosition)
        self.max_pool = MaxPool(kernel_size=(2, 2), stride=(2, 2))


    def forward(self, x):
        encoder_position_inputs, encoder_saliency_inputs, decoder_position_inputs, decoder_saliency_inputs = x
        
        # Encoding
        out_enc_pos, states_1 = self.sense_pos_enc(encoder_position_inputs)
        out_enc_sal, states_2 = self.sense_sal_enc(encoder_saliency_inputs)
        out_enc_sal = self.max_pool(out_enc_sal[0]).flatten(start_dim=2)
        out_enc_sal = self.fc_sal_enc(out_enc_sal)

        conc_out_enc = torch.cat([out_enc_sal, out_enc_pos], dim=-1)
        fuse_out_enc, states_fuse = self.fuse_1_enc(conc_out_enc)

        # Decoding
        all_pos_outputs = []
        inputs = decoder_position_inputs
        for curr_idx in range(self.h_window):
            out_enc_pos, states_1 = self.sense_pos_dec(inputs, states_1)

            selected_timestep_saliency = decoder_saliency_inputs[:, curr_idx:curr_idx+1]
            out_enc_sal, states_2 = self.sense_sal_dec(selected_timestep_saliency, states_2)
            out_enc_sal = self.max_pool(out_enc_sal[0]).flatten(start_dim=2)
            out_enc_sal = self.fc_sal_dec(out_enc_sal)

            conc_out_dec = torch.cat([out_enc_sal, out_enc_pos], dim=-1)

            fuse_out_dec_1, states_fuse = self.fuse_1_dec(conc_out_dec, states_fuse)
            fuse_out_dec_2 = self.fuse_2(fuse_out_dec_1)

            outputs_delta = self.fc_layer_out(fuse_out_dec_2)

            decoder_pred = self.to_position([inputs, outputs_delta])

            all_pos_outputs.append(decoder_pred)
            # Reinject the outputs as inputs for the next loop iteration as well as update the states
            inputs = decoder_pred

        # Concatenate all predictions
        decoder_outputs_pos = torch.cat(all_pos_outputs, dim=1)

        return decoder_outputs_pos


if __name__ == "__main__":
    B = 32
    M_WINDOW, H_WINDOW = 5, 25
    SALMAP_SHAPE = (64, 128)

    encoder_position_inputs = torch.randn(B, M_WINDOW, 3)
    encoder_saliency_inputs = torch.randn(B, M_WINDOW, 1, *SALMAP_SHAPE)
    decoder_position_inputs = torch.randn(B, 1, 3)
    decoder_saliency_inputs = torch.randn(B, H_WINDOW, 1, *SALMAP_SHAPE)
    
    model = TRACKConvLstmModel(M_WINDOW, H_WINDOW, SALMAP_SHAPE)
    
    decoder_outputs_pos = model([encoder_position_inputs, encoder_saliency_inputs, decoder_position_inputs, decoder_saliency_inputs])
    print(decoder_outputs_pos.shape)

    get_model_size(model)