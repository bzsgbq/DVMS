import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from models.models_utils import LambdaLayer, toPosition, get_model_size
    from models.sal_convs import SalConvs
except:
    from models_utils import LambdaLayer, toPosition, get_model_size
    from sal_convs import SalConvs


class ViewpointPredictionModel(nn.Module):
    def __init__(self, M_WINDOW, H_WINDOW, salmap_shape, lstm_hidden_size=64, lstm_num_layers=1, cnn_hidden_size=64, att_hidden_size=256):
        super(ViewpointPredictionModel, self).__init__()
        self.m_window = M_WINDOW
        self.h_window = H_WINDOW
        self.salmap_shape = salmap_shape
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.cnn_hidden_size = cnn_hidden_size
        self.att_hidden_size = att_hidden_size

        # LSTM for processing pos information
        self.pos_enc = nn.LSTM(input_size=3, hidden_size=self.lstm_hidden_size, num_layers=self.lstm_num_layers, batch_first=True)
        
        # CNN for processing sal information
        self.sense_sal = SalConvs(salmap_shape=self.salmap_shape, init_ch_size=16, output_size=self.cnn_hidden_size)

        self.sense_pos = nn.Sequential(
            nn.Linear(3, 2*self.lstm_hidden_size),
            nn.ReLU(),
            nn.Linear(2*self.lstm_hidden_size, self.lstm_hidden_size)
        )

        self.decoder = nn.LSTM(input_size=self.lstm_hidden_size+self.cnn_hidden_size, hidden_size=self.lstm_hidden_size, num_layers=self.lstm_num_layers, batch_first=True)
        
        # # Attention mechanism for dynamically combining pos and sal information
        # self.attention = nn.Sequential(
        #     nn.Linear(self.lstm_hidden_size + self.cnn_hidden_size, self.att_hidden_size),
        #     nn.Tanh(),
        #     nn.Linear(self.att_hidden_size, 1)
        # )
        
        # Final output layer for predicting future viewpoints
        self.output_layer = nn.Linear(self.lstm_hidden_size, 3)
        
        self.to_position = LambdaLayer(toPosition)

    def forward(self, x):
        pos, sal = x

        # Encoding
        _, state = self.pos_enc(pos)

        # Decoding
        all_pos_outputs = [pos[:, -1:, :]]
        
        for t in range(self.h_window):
            inputs = torch.cat([self.sense_pos(all_pos_outputs[-1]), self.sense_sal(sal[:, t]).unsqueeze(1)], dim=-1)
            out_dec_pos, state = self.decoder(inputs, state)
            outputs_delta = self.output_layer(out_dec_pos)
            decoder_pred = self.to_position([all_pos_outputs[-1], outputs_delta])
            all_pos_outputs.append(decoder_pred)

        # Concatenate all predictions
        decoder_outputs_pos = torch.cat(all_pos_outputs[1:], dim=1)

        return decoder_outputs_pos

        # Process pos information with LSTM
        pos_output, _ = self.pos_lstm(pos)
        
        # Process sal information with CNN
        sal = torch.squeeze(sal, dim=-1)
        sal_output = self.sal_conv(sal)
        
        # Concatenate pos and sal features along the time dimension
        combined_output = torch.cat([pos_output, sal_output], dim=1)
        
        # Compute attention weights based on combined features and use them to weight the pos and sal features
        attention_weights = F.softmax(self.attention(combined_output), dim=1)
        weighted_pos = torch.sum(attention_weights * pos_output, dim=1)
        weighted_sal = torch.sum(attention_weights * sal_output, dim=1)
        
        # Concatenate the weighted pos and sal features and use them to predict future viewpoints
        combined_output = torch.cat([weighted_pos, weighted_sal], dim=1)
        future_viewpoints = self.output_layer(combined_output)
        
        return future_viewpoints
    

if __name__ == '__main__':
    B = 8
    M_WINDOW = 5
    H_WINDOW = 25
    SALMAP_SHAPE = (256, 256)

    model = ViewpointPredictionModel(M_WINDOW=M_WINDOW, H_WINDOW=H_WINDOW, salmap_shape=SALMAP_SHAPE)
    pos = torch.randn(B, M_WINDOW, 3)
    sal = torch.randn(B, H_WINDOW, 1, *SALMAP_SHAPE)
    y = model([pos, sal])
    print(y.shape)

    get_model_size(model)