import torch
from torch import nn

from .models_utils import to_position_normalized
from .types_ import *


class DVMS(nn.Module):
    def __init__(self,
                 in_channels: int,
                 h_window: int = 25,
                 latent_dim: int = 128,
                 n_hidden: int = 2,
                 hidden_dim: int = 64,
                 n_samples_train: int = 5,
                 device: torch.device = torch.device('cuda')) -> None:
        super(DVMS, self).__init__()

        self.seq_len = h_window
        self.latent_dim = latent_dim
        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.n_samples_train = n_samples_train
        self.device = device
        self.fixed_points = torch.Tensor([[-(n_samples_train - 1) / 2 + i] for i in range(n_samples_train)]).float().to(device)

        self.past_encoder = nn.GRU(in_channels, hidden_dim, n_hidden, batch_first=True)
        self.past_decoder_bottleneck = nn.Linear(hidden_dim * n_hidden, self.latent_dim)
        self.decoder_future_state = nn.Linear(self.latent_dim + self.fixed_points.shape[-1], hidden_dim * n_hidden)
        self.decoder_future = nn.GRU(in_channels, hidden_dim, n_hidden, batch_first=True)
        self.fc_diff = nn.Linear(hidden_dim, in_channels)

    def encode(self, past: Tensor) -> Tensor:
        """
        Encode the past by passing through the encoder network and return an encoded past.
        :param past: Past trajectory to encode [batch_size x M x in_channels]
        :return: (Tensor) The encoded past
        """
        _, past_state = self.past_encoder(past)
        past_state = torch.flatten(torch.movedim(past_state, 0, 1), start_dim=1)
        return past_state

    def decode(self, inputs: List[Tensor]) -> Tensor:
        """
        Map the given latent codes onto the trajectory space.
        :param inputs: List of 3 Tensor objects containing:
            - past_state: (Tensor) Past encoded states to initialize decoder state [batch_size * n_samples_train x hidden_dim]
            - z: (Tensor) Fixed points from latent space to represent trajectory "mode" [batch_size * n_samples_train x 1]
            - current: (Tensor) Current coordinates to initialize decoder input [batch_size * n_samples_train x 1 x in_channels]
        :return: (Tensor) Decoded future trajectories [batch_size * n_samples_train x H x in_channels]
        """
        past_state, z, current = inputs

        combined_state = torch.cat([self.past_decoder_bottleneck(past_state),
                                    z], -1)
        hidden_state = self.decoder_future_state(combined_state)
        hidden_states = []
        for i in range(self.n_hidden):
            hidden_states.append(hidden_state[..., i * self.hidden_dim:(i + 1) * self.hidden_dim])
        hidden_states = torch.stack(hidden_states, dim=0)
        decoder_input = current
        result = []
        for _ in range(self.seq_len):
            result_t, hidden_states = self.decoder_future(decoder_input, hidden_states)
            result_delta = self.fc_diff(result_t.squeeze(1)).unsqueeze(1)
            result_pos = to_position_normalized([decoder_input, result_delta])
            result.append(result_pos)
            decoder_input = result_pos
        return torch.cat(result, dim=1)

    def forward(self, inputs: List[Tensor]) -> List[Tensor]:
        """
        Forward pass through the network. Encode the past trajectory and decode the future trajectory.
        :param inputs: List of 3 Tensor objects containing:
            - past: (Tensor) Past trajectories [batch_size x M x in_channels]
            - current: (Tensor) Current coordinates [batch_size x 1 x in_channels]
            - future: (Tensor) Ground truth future trajectories [batch_size * H x in_channels]
        :return: (List of Tensor) Decoded future trajectories [batch_size * n_samples_train x H x in_channels]
                                  along with aligned repeated ground truth futures [batch_size * n_samples_train x H x in_channels]
        """
        (past, current), future = inputs
        past_state = self.encode(past)

        past_state = torch.repeat_interleave(past_state, self.n_samples_train, dim=0)
        current = torch.repeat_interleave(current, self.n_samples_train, dim=0)
        future = torch.repeat_interleave(future, self.n_samples_train, dim=0)

        batch_size = past.shape[0]
        z = self.fixed_points.repeat(batch_size, 1)

        return [self.decode([past_state, z, current]), future]

    def loss_function(self, *args) -> dict:
        """
        Compute the variety loss function.
        :param args: The first two parameters must be the predicted and the ground truth future trajectories [batch_size * n_samples_train x H x in_channels]
        :return: (dict) Dictionary of all computed metrics
        """
        y_hat = args[0]
        y = args[1]

        distance = torch.linalg.norm(y - y_hat, dim=-1).reshape(-1, self.n_samples_train, self.seq_len)
        loss, _ = torch.min(torch.mean(distance, -1), -1)
        loss = torch.mean(loss)

        return {'loss': loss}

    def sample(self, inputs: List[Tensor]) -> Tensor:
        """
        Sample from the latent space and return the corresponding decoded trajectories.
        :param inputs: List of 2 Tensor objects containing:
            - past: (Tensor) Past trajectory to initialize decoder state [batch_size x M x in_channels]
            - current: (Tensor) Current coordinates to initialize decoder input [batch_size x 1 x in_channels]
        :return: (Tensor)
        """

        past, current = inputs
        batch_size = past.shape[0]

        z = self.fixed_points.repeat(batch_size, 1)

        past_state = self.encode(past)

        past_state = torch.repeat_interleave(past_state, self.n_samples_train, dim=0)
        current = torch.repeat_interleave(current, self.n_samples_train, dim=0)

        samples = self.decode([past_state, z, current])
        return samples
