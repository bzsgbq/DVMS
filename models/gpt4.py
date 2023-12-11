import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from models.models_utils import get_model_size
    from models.sal_convs import SalConvs
except:
    from models_utils import get_model_size
    from sal_convs import SalConvs


class ViewpointPredictor(nn.Module):
    def __init__(self, H_WINDOW, salmap_shape, lstm_hidden_size=64, cnn_hidden_size=64):
        super(ViewpointPredictor, self).__init__()
        self.h_window = H_WINDOW
        self.salmap_shape = salmap_shape
        self.lstm_hidden_size = lstm_hidden_size
        self.cnn_hidden_size = cnn_hidden_size

        # LSTM for pos information
        self.lstm = nn.LSTM(input_size=3, hidden_size=self.lstm_hidden_size, batch_first=True)
        
        # Convolutional layers for sal information
        self.convs = SalConvs(salmap_shape=self.salmap_shape, init_ch_size=16, output_size=self.cnn_hidden_size)
        
        self.fc1 = nn.Linear(lstm_hidden_size + cnn_hidden_size, 128)
        self.fc2 = nn.Linear(128, 3)
        # self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        pos, sal = x
        batch_size = pos.shape[0]

        # Process pos information with LSTM
        lstm_out, _ = self.lstm(pos)
        lstm_out = lstm_out[:, -1] # Take output of the last time step

        # Process sal information with CNN
        cnn_out = self.convs(sal)

        # Attention mechanism to adjust the weights of pos and sal
        lstm_out_expanded = lstm_out.unsqueeze(1).expand(-1, self.h_window, -1)
        combined_features = torch.cat([lstm_out_expanded, cnn_out], dim=2)
        combined_features = F.relu(self.fc1(combined_features))
        
        # Predict future viewpoints
        viewpoints = self.fc2(combined_features) # (batch_size, self.h_window, 3)
        return viewpoints


if __name__ == '__main__':
    B = 4
    M_WINDOW = 5
    H_WINDOW = 10
    SALMAP_SHAPE = (64, 128)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    pos = torch.rand(B, M_WINDOW, 3).float().to(device)
    sal = torch.rand(B, H_WINDOW, 1, *SALMAP_SHAPE).float().to(device)
    model = ViewpointPredictor(H_WINDOW=H_WINDOW, salmap_shape=SALMAP_SHAPE).float().to(device)
    output = model([pos, sal])
    print(output.shape) # Should be (batch_size, self.h_window, 3)

    get_model_size(model)

    # salmap_shape = (90, 160)
    # print(int(math.log(salmap_shape[0], 2)), int(math.log(salmap_shape[1], 2)))

    # x = 64
    # while x > 1:
    #     print(x)
    #     x = int(x // 2)