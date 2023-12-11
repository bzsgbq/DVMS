import torch
import torch.nn as nn

try:
    from models.models_utils import LambdaLayer, toPosition, get_model_size
except:
    from models_utils import LambdaLayer, toPosition, get_model_size


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 4, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(batch_size, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=-1)))
        attention = torch.softmax(self.v(energy), dim=1)
        context = attention.bmm(encoder_outputs)
        return context


class TRACKModel(nn.Module):
    def __init__(self, M_WINDOW, H_WINDOW, salmap_shape, hidden_size=256, num_layers=1):
        super(TRACKModel, self).__init__()
        self.m_window = M_WINDOW
        self.h_window = H_WINDOW
        self.salmap_height = salmap_shape[0]
        self.salmap_width = salmap_shape[1]
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.sense_pos_enc = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=1, dim_feedforward=self.hidden_size, batch_first=True)
        self.sense_sal_enc = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=1, dim_feedforward=self.hidden_size, batch_first=True)
        self.fuse_1_enc = nn.TransformerEncoderLayer(d_model=2*self.hidden_size, nhead=1, dim_feedforward=self.hidden_size, batch_first=True)
        self.sense_pos_dec = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=1, dim_feedforward=self.hidden_size, batch_first=True)
        self.sense_sal_dec = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=1, dim_feedforward=self.hidden_size, batch_first=True)
        self.fuse_1_dec = nn.TransformerEncoderLayer(d_model=2*self.hidden_size, nhead=1, dim_feedforward=self.hidden_size, batch_first=True)
        self.fuse_2 = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
        self.fc_layer_out = nn.Linear(in_features=self.hidden_size, out_features=3)

        self.to_position = LambdaLayer(toPosition)

        self.attention = Attention(self.hidden_size)
        self.fc_pos_enc = nn.Linear(in_features=3, out_features=self.hidden_size)
        self.fc_sal_enc = nn.Linear(in_features=self.salmap_height*self.salmap_width, out_features=self.hidden_size)
        self.fc_pos_dec = nn.Linear(in_features=3, out_features=self.hidden_size)
        self.fc_sal_dec = nn.Linear(in_features=self.salmap_height*self.salmap_width, out_features=self.hidden_size)


    def forward(self, x):
        encoder_position_inputs, encoder_saliency_inputs, decoder_position_inputs, decoder_saliency_inputs = x

        # Encoding
        out_enc_pos = self.sense_pos_enc(self.fc_pos_enc(encoder_position_inputs))
        out_enc_sal = self.sense_sal_enc(self.fc_sal_enc(encoder_saliency_inputs.flatten(start_dim=2)))
        conc_out_enc = torch.cat([out_enc_sal, out_enc_pos], dim=-1)
        # fuse_out_enc = self.fuse_1_enc(conc_out_enc.transpose(0, 1))

        # Decoding
        all_pos_outputs = []
        inputs = decoder_position_inputs
        for curr_idx in range(self.h_window):
            out_enc_pos = self.sense_pos_dec(self.fc_pos_dec(inputs))
            selected_timestep_saliency = decoder_saliency_inputs[:, curr_idx:curr_idx+1]
            flatten_timestep_saliency = selected_timestep_saliency.view(selected_timestep_saliency.shape[0], 1, -1)
            out_enc_sal = self.sense_sal_dec(self.fc_sal_dec(flatten_timestep_saliency))
            conc_out_dec = torch.cat([out_enc_sal, out_enc_pos], dim=-1)
            print(conc_out_dec.shape)
            exit()
            context = self.attention(conc_out_dec[-1], conc_out_enc)
            fuse_out_dec_1 = self.fuse_1_dec(torch.cat([conc_out_dec[-1], context], dim=-1).unsqueeze(0))
            fuse_out_dec_2 = self.fuse_2(fuse_out_dec_1.squeeze(0))
            outputs_delta = self.fc_layer_out(fuse_out_dec_2)
            decoder_pred = self.to_position([inputs, outputs_delta])
            all_pos_outputs.append(decoder_pred)
            inputs = decoder_pred

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
    
    model = TRACKModel(M_WINDOW, H_WINDOW, SALMAP_SHAPE)
    
    decoder_outputs_pos = model([encoder_position_inputs, encoder_saliency_inputs, decoder_position_inputs, decoder_saliency_inputs])
    print(decoder_outputs_pos.shape)

    get_model_size(model)