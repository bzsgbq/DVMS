import torch
import torch.nn as nn
import torch.optim as optim

try:
    from models.models_utils import LambdaLayer, toPosition, get_model_size
except:
    from models_utils import LambdaLayer, toPosition, get_model_size


# class TRACKModel(nn.Module):
#     def __init__(self, M_WINDOW, H_WINDOW, salmap_shape, hidden_size=256, num_layers=1):
#         super(TRACKModel, self).__init__()
#         self.m_window = M_WINDOW
#         self.h_window = H_WINDOW
#         self.salmap_height = salmap_shape[0]
#         self.salmap_width = salmap_shape[1]
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers

#         self.pos_encoder = nn.LSTM(input_size=3, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
#         self.sal_encoder = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(self.salmap_height*self.salmap_width*128, self.hidden_size)
#         )
        
#         self.fuse_encoder = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        
#         self.pos_decoder = nn.LSTM(input_size=3, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
#         self.sal_decoder = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(self.salmap_height*self.salmap_width*128, self.hidden_size)
#         )
        
#         self.fuse_decoder = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        
#         self.fuse = nn.Linear(in_features=2*self.hidden_size, out_features=self.hidden_size)
#         self.fc_layer_out = nn.Linear(in_features=self.hidden_size, out_features=3)
        
#         self.to_position = LambdaLayer(toPosition)
#         self.softmax = nn.Softmax(dim=-1)
        
#     def forward(self, x):
#         encoder_position_inputs, encoder_saliency_inputs, decoder_position_inputs, decoder_saliency_inputs = x
        
#         batch_size = encoder_position_inputs.size(0)

#         # Encoding
#         encoded_pos, (last_hidden_pos, _) = self.pos_encoder(encoder_position_inputs)
#         encoded_sal = self.sal_encoder(torch.flatten(encoder_saliency_inputs, start_dim=0, end_dim=1)).view(batch_size, self.m_window, self.hidden_size)
#         encoded_sal, (last_hidden_sal, _) = self.fuse_encoder(encoded_sal)

#         # Self-Attention Mechanism
#         pos_query = last_hidden_pos[-1].unsqueeze(1).repeat(1, self.m_window, 1)
#         att_pos = self.softmax(torch.bmm(encoded_pos, pos_query.transpose(1, 2)))
#         att_pos = torch.bmm(att_pos.transpose(1, 2), encoded_pos)

#         sal_query = last_hidden_sal[-1].unsqueeze(1).repeat(1, self.m_window, 1)
#         att_sal = self.softmax(torch.bmm(encoded_sal, sal_query.transpose(1, 2)))
#         att_sal = torch.bmm(att_sal.transpose(1, 2), encoded_sal)

#         att_fuse = torch.cat([att_pos, att_sal], dim=-1)  # (batch_size, m_window, 2*hidden_size)
#         att_fuse, (last_hidden_fuse, _) = self.fuse_decoder(att_fuse)
#         # print(last_hidden_fuse.size())
#         # exit()
#         last_hidden_fuse = last_hidden_fuse.squeeze(0)

#         # Decoding
#         all_pos_outputs = []
#         inputs = decoder_position_inputs
#         for curr_idx in range(self.h_window):
#             encoded_pos, (last_hidden_pos, _) = self.pos_decoder(inputs, (last_hidden_fuse.unsqueeze(0), last_hidden_fuse.unsqueeze(0)))

#             selected_timestep_saliency = decoder_saliency_inputs[:, curr_idx:curr_idx+1]
#             encoded_sal = self.sal_decoder(torch.flatten(selected_timestep_saliency, start_dim=0, end_dim=1)).view(batch_size, 1, self.hidden_size)
#             encoded_sal, (last_hidden_sal, _) = self.fuse_decoder(encoded_sal, (last_hidden_fuse.unsqueeze(0), last_hidden_fuse.unsqueeze(0)))

#             att_pos = self.softmax(torch.bmm(encoded_pos, last_hidden_pos[-1].unsqueeze(2))).squeeze(2)
#             att_sal = self.softmax(torch.bmm(encoded_sal, last_hidden_sal[-1].unsqueeze(2))).squeeze(2)

#             att_fuse = torch.cat([att_pos, att_sal], dim=-1)
#             att_fuse = self.fuse(att_fuse)
#             att_fuse = torch.relu(att_fuse)
#             att_fuse, (last_hidden_fuse, _) = self.fuse_decoder(att_fuse.unsqueeze(1), (last_hidden_fuse.unsqueeze(0), last_hidden_fuse.unsqueeze(0)))
#             last_hidden_fuse = last_hidden_fuse.squeeze(0)

#             outputs_delta = self.fc_layer_out(att_fuse)
#             decoder_pred = self.to_position([inputs, outputs_delta])

#             all_pos_outputs.append(decoder_pred)
#             inputs = decoder_pred

#         decoder_outputs_pos = torch.cat(all_pos_outputs, dim=1)

#         return decoder_outputs_pos


# class TRACKSageModel(nn.Module):
#     def __init__(self, M_WINDOW, H_WINDOW, salmap_shape, hidden_size=256, num_layers=1):
#         super(TRACKSageModel, self).__init__()
#         self.m_window = M_WINDOW
#         self.h_window = H_WINDOW
#         self.salmap_height = salmap_shape[0]
#         self.salmap_width = salmap_shape[1]
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers

#         self.sense_pos_enc = nn.LSTM(input_size=3, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
#         self.sense_sal_enc = nn.LSTM(input_size=self.salmap_height*self.salmap_width, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
#         self.fuse_1_enc = nn.LSTM(input_size=2*self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)

#         self.attn_pos_enc = nn.MultiheadAttention(self.hidden_size, num_heads=8)
#         self.attn_sal_enc = nn.MultiheadAttention(self.hidden_size, num_heads=8)
#         self.attn_fuse_enc = nn.MultiheadAttention(2*self.hidden_size, num_heads=8)

#         self.sense_pos_dec = nn.LSTM(input_size=3, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
#         self.sense_sal_dec = nn.LSTM(input_size=self.salmap_height*self.salmap_width, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
#         self.fuse_1_dec = nn.LSTM(input_size=2*self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)

#         self.attn_pos_dec = nn.MultiheadAttention(self.hidden_size, num_heads=8)
#         self.attn_sal_dec = nn.MultiheadAttention(self.hidden_size, num_heads=8)
#         self.attn_fuse_dec = nn.MultiheadAttention(2*self.hidden_size, num_heads=8)

#         self.fuse_2 = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
#         self.fc_layer_out = nn.Linear(in_features=self.hidden_size, out_features=3)
#         self.to_position = LambdaLayer(toPosition)

#     def forward(self, x):
#         encoder_position_inputs, encoder_saliency_inputs, decoder_position_inputs, decoder_saliency_inputs = x
        
#         # Encoding
#         out_enc_pos, states_1 = self.sense_pos_enc(encoder_position_inputs)
#         out_flat_enc = encoder_saliency_inputs.flatten(start_dim=2)
#         out_enc_sal, states_2 = self.sense_sal_enc(out_flat_enc)
#         conc_out_enc = torch.cat([out_enc_sal, out_enc_pos], dim=-1)
#         fuse_out_enc, states_fuse = self.fuse_1_enc(conc_out_enc)

#         # Attention-based Encoding
#         attn_out_pos_enc, _ = self.attn_pos_enc(out_enc_pos.permute(1, 0, 2), out_enc_pos.permute(1, 0, 2), out_enc_pos.permute(1, 0, 2))
#         attn_out_sal_enc, _ = self.attn_sal_enc(out_enc_sal.permute(1, 0, 2), out_enc_sal.permute(1, 0, 2), out_enc_sal.permute(1, 0, 2))
#         attn_out_fuse_enc, _ = self.attn_fuse_enc(conc_out_enc.permute(1, 0, 2), conc_out_enc.permute(1, 0, 2), conc_out_enc.permute(1, 0, 2))

#         # Decoding
#         all_pos_outputs = []
#         inputs = decoder_position_inputs
#         for curr_idx in range(self.h_window):
#             out_enc_pos, states_1 = self.sense_pos_dec(inputs, states_1)
#             selected_timestep_saliency = decoder_saliency_inputs[:, curr_idx:curr_idx+1]
#             flatten_timestep_saliency = selected_timestep_saliency.view(selected_timestep_saliency.shape[0], 1, -1)
#             out_enc_sal, states_2 = self.sense_sal_dec(flatten_timestep_saliency, states_2)

#             # Attention-based Decoding
#             attn_out_pos_dec, _ = self.attn_pos_dec(out_enc_pos.permute(1, 0, 2), attn_out_pos_enc, attn_out_pos_enc)
#             attn_out_sal_dec, _ = self.attn_sal_dec(out_enc_sal.permute(1, 0, 2), attn_out_sal_enc, attn_out_sal_enc)
#             conc_out_dec = torch.cat([attn_out_sal_dec.permute(1, 0, 2), attn_out_pos_dec.permute(1, 0, 2)], dim=-1)
#             attn_out_fuse_dec, _ = self.attn_fuse_dec(conc_out_dec.permute(1, 0, 2), attn_out_fuse_enc, attn_out_fuse_enc)

#             fuse_out_dec_1, states_fuse = self.fuse_1_dec(attn_out_fuse_dec.permute(1, 0, 2), states_fuse)
#             fuse_out_dec_2 = self.fuse_2(fuse_out_dec_1)
#             outputs_delta = self.fc_layer_out(fuse_out_dec_2)

#             decoder_pred = self.to_position([inputs, outputs_delta])
#             all_pos_outputs.append(decoder_pred)

#             # Reinject the outputs as inputs for the next loop iteration as well as update the states
#             inputs = decoder_pred

#         # Concatenate all predictions
#         decoder_outputs_pos = torch.cat(all_pos_outputs, dim=1)

#         return decoder_outputs_pos



from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TRACKSageModel(nn.Module):
    def __init__(self, M_WINDOW, H_WINDOW, salmap_shape, hidden_size=256, num_layers=1, dropout=0.1):
        super(TRACKSageModel, self).__init__()
        self.m_window = M_WINDOW
        self.h_window = H_WINDOW
        self.salmap_height = salmap_shape[0]
        self.salmap_width = salmap_shape[1]
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.pos_enc = nn.Linear(3, hidden_size)
        self.sal_enc = nn.Linear(self.salmap_height*self.salmap_width, hidden_size)
        self.fuse_enc = nn.Linear(2*hidden_size, hidden_size)

        self.pos_dec = nn.Linear(3, hidden_size)
        self.sal_dec = nn.Linear(self.salmap_height*self.salmap_width, hidden_size)
        self.fuse_dec = nn.Linear(2*hidden_size, hidden_size)

        self.fc_layer_out = nn.Linear(hidden_size, 3)

        encoder_layer = TransformerEncoderLayer(d_model=hidden_size, nhead=8, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = TransformerEncoderLayer(d_model=hidden_size, nhead=8, dropout=dropout)
        self.transformer_decoder = TransformerEncoder(decoder_layer, num_layers=num_layers)

        self.to_position = LambdaLayer(toPosition)

    def forward(self, x):
        # unpack inputs
        encoder_position_inputs, encoder_saliency_inputs, decoder_position_inputs, decoder_saliency_inputs = x

        # # Encoding
        # out_enc_pos = self.pos_enc(encoder_position_inputs)
        # out_flat_enc = encoder_saliency_inputs.flatten(start_dim=2)
        # out_enc_sal = self.sal_enc(out_flat_enc)
        # conc_out_enc = torch.cat([out_enc_sal, out_enc_pos], dim=-1)
        # fuse_out_enc = self.fuse_enc(conc_out_enc)
        # encoder_output = self.transformer_encoder(fuse_out_enc.transpose(0, 1))

        # Decoding
        all_pos_outputs = []
        inputs = decoder_position_inputs
        for curr_idx in range(self.h_window):
            out_enc_pos = self.pos_dec(inputs)
            selected_timestep_saliency = decoder_saliency_inputs[:, curr_idx:curr_idx+1]
            flatten_timestep_saliency = selected_timestep_saliency.view(selected_timestep_saliency.shape[0], 1, -1)
            out_enc_sal = self.sal_dec(flatten_timestep_saliency)
            conc_out_dec = torch.cat([out_enc_sal, out_enc_pos], dim=-1)
            fuse_out_dec = self.fuse_dec(conc_out_dec)

            # apply transformer decoder with attention mask
            decoder_input = fuse_out_dec.transpose(0, 1)
            attention_mask = self._generate_square_subsequent_mask(decoder_input.shape[0])
            decoder_output = self.transformer_decoder(decoder_input, mask=attention_mask)

            # compute position prediction
            outputs_delta = self.fc_layer_out(decoder_output[-1])
            decoder_pred = self.to_position([inputs, outputs_delta])

            all_pos_outputs.append(decoder_pred)
            # Reinject the outputs as inputs for the next loop iteration
            inputs = decoder_pred

        # Concatenate all predictions
        decoder_outputs_pos = torch.cat(all_pos_outputs, dim=1)

        return decoder_outputs_pos

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


if __name__ == "__main__":
    B = 32
    M_WINDOW, H_WINDOW = 5, 25
    SALMAP_SHAPE = (64, 128)

    encoder_position_inputs = torch.randn(B, M_WINDOW, 3)
    encoder_saliency_inputs = torch.randn(B, M_WINDOW, 1, *SALMAP_SHAPE)
    decoder_position_inputs = torch.randn(B, 1, 3)
    decoder_saliency_inputs = torch.randn(B, H_WINDOW, 1, *SALMAP_SHAPE)
    
    model = TRACKSageModel(M_WINDOW, H_WINDOW, SALMAP_SHAPE)
    
    decoder_outputs_pos = model([encoder_position_inputs, encoder_saliency_inputs, decoder_position_inputs, decoder_saliency_inputs])
    print(decoder_outputs_pos.shape)

    get_model_size(model)