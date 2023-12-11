import torch
import torch.nn as nn
import torch.optim as optim

try:
    from models.models_utils import LambdaLayer, toPosition, get_model_size
    from models.sal_convs import SalConvs
except:
    from models_utils import LambdaLayer, toPosition, get_model_size
    from sal_convs import SalConvs

USE_CNN = False


class TRACKLeftModel(nn.Module):
    def __init__(self, M_WINDOW, H_WINDOW, salmap_shape, hidden_size=256, num_layers=1):
        super(TRACKLeftModel, self).__init__()
        self.m_window = M_WINDOW
        self.h_window = H_WINDOW
        self.salmap_shape = salmap_shape
        self.salmap_height = salmap_shape[0]
        self.salmap_width = salmap_shape[1]
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # self.sense_pos_enc = nn.LSTM(input_size=3, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        # self.sense_sal_enc = nn.LSTM(input_size=self.salmap_height*self.salmap_width, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fuse_1_enc = nn.LSTM(input_size=2*self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        # self.sense_pos_dec = nn.LSTM(input_size=3, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        # self.sense_sal_dec = nn.LSTM(input_size=self.salmap_height*self.salmap_width, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fuse_1_dec = nn.LSTM(input_size=2*self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fuse_2 = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
        self.fc_layer_out = nn.Linear(in_features=self.hidden_size, out_features=3)
        
        self.to_position = LambdaLayer(toPosition)
        self.fc_pos_enc = nn.Linear(in_features=3, out_features=self.hidden_size)
        self.fc_pos_dec = nn.Linear(in_features=3, out_features=self.hidden_size)
        # self.fc_pos_dec = self.fc_pos_enc
        if USE_CNN:
            self.sal_convs_enc = SalConvs(salmap_shape=self.salmap_shape, init_ch_size=16, output_size=self.hidden_size, num_conv_layer=3)
            self.sal_convs_dec = SalConvs(salmap_shape=self.salmap_shape, init_ch_size=16, output_size=self.hidden_size, num_conv_layer=3)
        else:
            self.fc_sal_enc = nn.Linear(in_features=self.salmap_height*self.salmap_width, out_features=self.hidden_size)
            self.fc_sal_dec = nn.Linear(in_features=self.salmap_height*self.salmap_width, out_features=self.hidden_size)
            # self.fc_sal_dec = self.fc_sal_enc

    def forward(self, x):
        encoder_position_inputs, encoder_saliency_inputs, decoder_position_inputs, decoder_saliency_inputs = x
        
        # Encoding
        # out_enc_pos, states_1 = self.sense_pos_enc(encoder_position_inputs)
        out_enc_pos = self.fc_pos_enc(encoder_position_inputs)

        if not USE_CNN:
            out_flat_enc = encoder_saliency_inputs.flatten(start_dim=2)
            # out_enc_sal, states_2 = self.sense_sal_enc(out_flat_enc)
            out_enc_sal = self.fc_sal_enc(out_flat_enc)
        else:
            out_enc_sal = self.sal_convs_enc(encoder_saliency_inputs)

        conc_out_enc = torch.cat([out_enc_sal, out_enc_pos], dim=-1)
        fuse_out_enc, states_fuse = self.fuse_1_enc(conc_out_enc)

        # Decoding
        all_pos_outputs = []
        inputs = decoder_position_inputs
        for curr_idx in range(self.h_window):
            # out_enc_pos, states_1 = self.sense_pos_dec(inputs, states_1)
            out_enc_pos = self.fc_pos_dec(inputs)

            selected_timestep_saliency = decoder_saliency_inputs[:, curr_idx:curr_idx+1]
            if not USE_CNN:
                flatten_timestep_saliency = selected_timestep_saliency.view(selected_timestep_saliency.shape[0], 1, -1)
                # out_enc_sal, states_2 = self.sense_sal_dec(flatten_timestep_saliency, states_2)
                out_enc_sal = self.fc_sal_dec(flatten_timestep_saliency)
            else:
                out_enc_sal = self.sal_convs_dec(selected_timestep_saliency)

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
    
    model = TRACKLeftModel(M_WINDOW, H_WINDOW, SALMAP_SHAPE)
    
    decoder_outputs_pos = model([encoder_position_inputs, encoder_saliency_inputs, decoder_position_inputs, decoder_saliency_inputs])
    print(decoder_outputs_pos.shape)

    get_model_size(model)