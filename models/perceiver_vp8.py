from transformers.models.perceiver.modeling_perceiver import *
from transformers.models.perceiver.configuration_perceiver import PerceiverConfig

try:
    from models.models_utils import get_model_size, get_xyz_grid, salmap2posalfeat
    from models.perceiver_vp import PerceiverVPPreprocessor, PerceiverVPDecoder, NUM_BANDS
except:
    from models_utils import get_model_size, get_xyz_grid, salmap2posalfeat
    from perceiver_vp import PerceiverVPPreprocessor, PerceiverVPDecoder, NUM_BANDS


'''
version 8: 在version5的基础上, 在pos之后增加一个可训练的参数; 这样一来, pos的维度就可以和sal的维度一样了.
(1) 当USE_FC=True时, 原始的pos维度为3, sal维度为4; 先在pos之后增加一个可训练的参数, 这样一来pos和sal的维度都变成了4; 然后通过fc_layer将pos和sal的维度都映射到HIDDEN_SIZE维度, 再position encoding;
(2) 当USE_FC=False时, 原始的pos维度为3, sal维度为N*4; 先在pos之后增加一个可训练的参数, 再 expand N 次, 这样一来pos和sal的维度就都变成了N*4; 然后position encoding; 然后直接送入perceiver io;

TODO: transformers.models.perceiver.modeling_perceiver.PerceiverModel.post_init会触发警告:
/usr/local/anaconda3/envs/gbq_pytorch/lib/python3.8/site-packages/torch/nn/modules/container.py:587: UserWarning: Setting attributes on ParameterDict is not supported.
  warnings.warn("Setting attributes on ParameterDict is not supported.")
'''


USE_FC = False
if USE_FC:
    HIDDEN_SIZE = 4 * NUM_BANDS
else:
    NUM_DATA = None  # saliency点的数量;


class PerceiverPosPreprocessor(AbstractPreprocessor):
    """
    Viewpoints preprocessing for Perceiver Encoder.

    Args:
        config ([*PerceiverConfig*]):
            Model configuration.
        prep_type (`str`, *optional*, defaults to `"patches"`):
            Preprocessor type to use. Only "patches" is supported.
        samples_per_patch (`int`, *optional*, defaults to 96):
            Number of samples per patch.
        position_encoding_type (`str`, *optional*, defaults to `"fourier"`):
            Type of position encoding to use. Can be "trainable" or "fourier".
        concat_or_add_pos (`str`, *optional*, defaults to `"concat"`):
            How to concatenate the position encoding to the input. Can be "concat" or "add".
        out_channels (`int`, *optional*, defaults to 64):
            Number of channels in the output.
        project_pos_dim (`int`, *optional*, defaults to -1):
            Dimension of the position encoding to project to. If -1, no projection is applied.
        **position_encoding_kwargs (`Dict`, *optional*):
            Keyword arguments for the position encoding.
    """

    def __init__(
        self,
        config,
        m_window: int = 5,
        h_window: int = 25,
        feat_dim: int = 255,
        position_encoding_type: str = "fourier",
        concat_or_add_pos: str = "concat",
        out_channels=64,
        project_pos_dim=-1,
        # **position_encoding_kwargs,
    ):
        super().__init__()
        self.config = config

        if concat_or_add_pos not in ["concat", "add"]:
            raise ValueError(f"Concat_or_pos {concat_or_add_pos} is invalid, can only be 'concat' or 'add'.")

        self.m_window = m_window
        self.h_window = h_window
        self.feat_dim = feat_dim

        self.position_encoding_type = position_encoding_type
        self.concat_or_add_pos = concat_or_add_pos
        self.project_pos_dim = project_pos_dim

        # Position embeddings
        self.position_embeddings, self.positions_projection = build_position_encoding(
            position_encoding_type=position_encoding_type,
            fourier_position_encoding_kwargs={
                "num_bands": NUM_BANDS,
                "max_resolution": (self.m_window+self.h_window,),
                "sine_only": False,
                "concat_pos": True,
            },
            trainable_position_encoding_kwargs={
                "num_channels": 2*NUM_BANDS+1, 
                "index_dims": self.m_window+self.h_window,
            },
            out_channels=out_channels,
            project_pos_dim=project_pos_dim,
        )

        self.trainable_pos_encoder = PerceiverTrainablePositionEncoding(index_dims=self.m_window, num_channels=1)

        self.fc_layer = nn.Linear(4, HIDDEN_SIZE) if USE_FC else nn.Identity()

    @property
    def num_channels(self) -> int:
        # position embedding
        if self.project_pos_dim > 0:
            pos_dim = self.project_pos_dim
        else:
            pos_dim = self.position_embeddings.output_size()
        if self.concat_or_add_pos == "add":
            return pos_dim
        return self.feat_dim + pos_dim

    def _build_network_inputs(self, inputs):
        """Construct the final input, including position encoding."""
        batch_size = inputs.shape[0]
        index_dims = inputs.shape[1:-1]

        # 为 inputs 添加 pos_trainable
        pos_trainable = self.trainable_pos_encoder(batch_size)#*index_dims[0]).view(batch_size, index_dims[0], -1)
        inputs = torch.cat([inputs, pos_trainable], dim=-1)

        # 根据USE_FC确定是否使用fc_layer将inputs从feat_dim维度映射到HIDDEN_SIZE维度
        inputs = self.fc_layer(inputs)
        
        if not USE_FC:  # inputs.shape = (B, T, 4)
            # (B, T, 4) -> (B, T, N, 4) -> (B, T, N*4)
            # inputs = inputs.unsqueeze(dim=2).repeat(1, 1, NUM_DATA, 1).flatten(start_dim=2)
            inputs = inputs.unsqueeze(dim=2).expand(-1, -1, NUM_DATA, -1).flatten(start_dim=2)

        # Construct the position encoding.
        if self.position_encoding_type == "trainable":
            pos_enc = self.position_embeddings(batch_size)[:, :index_dims[0]]
        elif self.position_encoding_type == "fourier":
            pos_enc = self.position_embeddings(index_dims, batch_size, device=inputs.device, dtype=inputs.dtype)

        # Optionally project them to a target dimension.
        pos_enc = self.positions_projection(pos_enc)

        if self.concat_or_add_pos == "concat":
            inputs_with_pos = torch.cat([inputs, pos_enc], dim=-1)
        elif self.concat_or_add_pos == "add":
            inputs_with_pos = inputs + pos_enc
        
        return inputs_with_pos, inputs

    def forward(self, inputs: torch.Tensor, pos: Optional[torch.Tensor] = None, network_input_is_1d: bool = True):
        inputs, inputs_without_pos = self._build_network_inputs(inputs)
        modality_sizes = None  # Size for each modality, only needed for multimodal
        return inputs, modality_sizes, inputs_without_pos


class PerceiverSalPreprocessor(AbstractPreprocessor):
    """
    Salmaps preprocessing for PosalFeat Encoder.

    Args:
        config ([*PerceiverConfig*]):
            Model configuration.
        prep_type (`str`, *optional*, defaults to `"patches"`):
            Preprocessor type to use. Only "patches" is supported.
        samples_per_patch (`int`, *optional*, defaults to 96):
            Number of samples per patch.
        position_encoding_type (`str`, *optional*, defaults to `"fourier"`):
            Type of position encoding to use. Can be "trainable" or "fourier".
        concat_or_add_pos (`str`, *optional*, defaults to `"concat"`):
            How to concatenate the position encoding to the input. Can be "concat" or "add".
        out_channels (`int`, *optional*, defaults to 64):
            Number of channels in the output.
        project_pos_dim (`int`, *optional*, defaults to -1):
            Dimension of the position encoding to project to. If -1, no projection is applied.
        **position_encoding_kwargs (`Dict`, *optional*):
            Keyword arguments for the position encoding.
    """

    def __init__(
        self,
        config,
        m_window: int,
        h_window: int,
        feat_dim: int,
        position_encoding_type: str = "fourier",
        concat_or_add_pos: str = "concat",
        out_channels=64,
        project_pos_dim=-1,
        # **position_encoding_kwargs,
    ):
        super().__init__()
        self.config = config

        if concat_or_add_pos not in ["concat", "add"]:
            raise ValueError(f"Concat_or_pos {concat_or_add_pos} is invalid, can only be 'concat' or 'add'.")

        self.m_window = m_window
        self.h_window = h_window
        self.feat_dim = feat_dim

        self.position_encoding_type = position_encoding_type
        self.concat_or_add_pos = concat_or_add_pos
        self.project_pos_dim = project_pos_dim

        # Position embeddings
        self.position_embeddings, self.positions_projection = build_position_encoding(
            position_encoding_type=position_encoding_type,
            fourier_position_encoding_kwargs={
                "num_bands": NUM_BANDS,
                "max_resolution": (self.m_window+self.h_window,),
                "sine_only": False,
                "concat_pos": True,
            },
            trainable_position_encoding_kwargs={
                "num_channels": 2*NUM_BANDS+1, 
                "index_dims": self.m_window+self.h_window,
            },
            out_channels=out_channels,
            project_pos_dim=project_pos_dim,
        )

        self.fc_layer = nn.Linear(4, HIDDEN_SIZE) if USE_FC else nn.Identity()

    @property
    def num_channels(self) -> int:
        # position embedding
        if self.project_pos_dim > 0:
            pos_dim = self.project_pos_dim
        else:
            pos_dim = self.position_embeddings.output_size()
        if self.concat_or_add_pos == "add":
            return pos_dim
        return self.feat_dim + pos_dim

    def _build_network_inputs(self, inputs):  # inputs.shape = (B, T, N, 4); return.shape = (B, T*N, 4+pos_dim)
        """Construct the final input, including position encoding."""
        batch_size, T, N = inputs.shape[:3]
        index_dims = inputs.shape[1:-1]

        # 根据USE_FC确定是否使用fc_layer将inputs从feat_dim维度映射到HIDDEN_SIZE维度
        inputs = self.fc_layer(inputs)

        # Construct the position encoding.
        if self.position_encoding_type == "trainable":
            pos_enc = self.position_embeddings(batch_size)[:, :T]
        elif self.position_encoding_type == "fourier":
            pos_enc = self.position_embeddings(index_dims, batch_size, device=inputs.device, dtype=inputs.dtype)

        # Optionally project them to a target dimension.
        pos_enc = self.positions_projection(pos_enc)  # (B, T, pos_dim)

        if USE_FC:
            # pos_enc = pos_enc.repeat(1, 1, N).view(batch_size, T*N, -1)  # (B, T*N, pos_dim)
            inputs = inputs.view(batch_size, T*N, -1)  # (B, T*N, 4)
        else:
            inputs = inputs.flatten(start_dim=2)  # (B, T, N*4)

        if self.concat_or_add_pos == "concat":
            inputs_with_pos = torch.cat([inputs, pos_enc], dim=-1)
        elif self.concat_or_add_pos == "add":
            inputs_with_pos = inputs + pos_enc

        return inputs_with_pos, inputs

    def forward(self, inputs: torch.Tensor, pos: Optional[torch.Tensor] = None, network_input_is_1d: bool = True):
        inputs, inputs_without_pos = self._build_network_inputs(inputs)
        modality_sizes = None  # Size for each modality, only needed for multimodal
        return inputs, modality_sizes, inputs_without_pos


class PerceiverForViewpointPrediction8(PerceiverPreTrainedModel):
    def __init__(self, config, m_window, h_window):
        super().__init__(config)
        self.m_window = m_window
        self.h_window = h_window

        self.perceiver = PerceiverModel(
            config,
            input_preprocessor=PerceiverMultimodalPreprocessor(
                min_padding_size=7 if not USE_FC else 6,
                modalities={
                    "pos": PerceiverPosPreprocessor(
                        config,
                        m_window=self.m_window,
                        h_window=self.h_window,
                        feat_dim=NUM_DATA*4 if not USE_FC else HIDDEN_SIZE,
                        position_encoding_type="fourier",
                        concat_or_add_pos="concat",
                    ),
                    "sal": PerceiverSalPreprocessor(
                        config,
                        m_window=self.m_window,
                        h_window=self.h_window,
                        feat_dim=NUM_DATA*4 if not USE_FC else HIDDEN_SIZE,
                        position_encoding_type="fourier",
                        concat_or_add_pos="concat",
                    ),
                },
                mask_probs={"pos": 0.0, "sal": 0.0},
            ),
            decoder=PerceiverVPDecoder(
                config,
                output_index_dims=self.m_window+self.h_window,
                num_channels=2*NUM_BANDS+1,
                output_num_channels=3,
                position_encoding_type="fourier",
                fourier_position_encoding_kwargs={
                    "num_bands": NUM_BANDS,
                    "max_resolution": (self.m_window+self.h_window,),
                    "sine_only": False,
                    "concat_pos": True,
                },
            ),
        )

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        inputs: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        subsampled_output_points: Optional[Dict[str, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, PerceiverClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.perceiver(
            inputs=inputs,
            attention_mask=attention_mask,
            subsampled_output_points=subsampled_output_points,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits if return_dict else outputs[0]

        loss = None
        if labels is not None:
            raise NotImplementedError("Multimodal autoencoding training is not yet supported")

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return PerceiverClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )
    


class PerceiverVPModel8(nn.Module):
    def __init__(self, config, M_WINDOW, H_WINDOW, salmap_shape):
        super(PerceiverVPModel8, self).__init__()
        self.m_window = M_WINDOW
        self.h_window = H_WINDOW
        global NUM_DATA
        NUM_DATA = salmap_shape[0]

        self.perceiver = PerceiverForViewpointPrediction8(config, self.m_window, self.h_window)


    def forward(self, x):
        enc_pos_in, m_sal_in, h_sal_in = x
        enc_pos_in = enc_pos_in[:, :, :3]# if self.pos_dim == 3 else enc_pos_in[:, :, -2:]
        enc_sal_in = torch.cat([m_sal_in, h_sal_in], dim=1)
        if not USE_FC:
            enc_sal_in = enc_sal_in.flatten(start_dim=2)
        inputs = dict(pos=enc_pos_in, sal=enc_sal_in)
        outputs = self.perceiver(inputs)
        return outputs.logits


    def predict(self, x):
        return self.forward(x)[:, -self.h_window:, :]


if __name__ == "__main__":
    cfg = PerceiverConfig(
        num_latents=256,
        d_latents=256,
        num_blocks=2,
        num_self_attends_per_block=8,
        num_self_attention_heads=8,
        num_cross_attention_heads=8,
        qk_channels=None,
        v_channels=None,
        cross_attention_shape_for_attention="kv",
        self_attention_widening_factor=1,
        cross_attention_widening_factor=1,
        hidden_act="gelu",
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        use_query_residual=True,
    )

    batch_size = 128
    m_window = 15
    h_window = 25
    salmap_shape = (512, 4)
    
    model = PerceiverVPModel8(cfg, m_window, h_window, salmap_shape)
    pos = torch.ones(batch_size, m_window, 3)
    m_sal = 2 * torch.ones(batch_size, m_window, NUM_DATA, 4)
    h_sal = 2 * torch.ones(batch_size, h_window, NUM_DATA, 4)
    inputs = (pos, m_sal, h_sal)
    outputs = model(inputs)
    print(outputs.shape)
    print('lll')
    get_model_size(model)
    print('lll')


    # # 测试PerceiverVPPreprocessor
    # feat_dim = 255
    # vp_preprocessor = PerceiverVPPreprocessor(
    #     cfg,
    #     m_window=5,
    #     h_window=25,
    #     feat_dim=feat_dim,
    #     position_encoding_type="fourier",
    #     concat_or_add_pos="concat",
    #     # out_channels=255,
    #     # project_pos_dim=256,
    # )
    # inputs = torch.rand(4, 30, feat_dim)
    # new_inputs, _, new_inputs_without_pos = vp_preprocessor(inputs)
    # print(new_inputs.shape)
    # print(new_inputs_without_pos.shape)
    # # 判断inputs和new_inputs_without_pos是否相等:
    # print(torch.all(torch.eq(inputs, new_inputs_without_pos)))
    # # 判断inputs和new_inputs的前半部分是否相等:
    # print(torch.all(torch.eq(inputs[:, :, :feat_dim], new_inputs[:, :, :feat_dim])))


    # # 测试PerceiverForViewpointPrediction
    # perceiver = PerceiverForViewpointPrediction(cfg, m_window=5, h_window=25, feat_dim=feat_dim)
    # outputs = perceiver(inputs)
    # print(outputs.logits.shape)
    # get_model_size(perceiver)


    # from transformers import PerceiverTokenizer
    # config = PerceiverConfig()
    # preprocessor = PerceiverTextPreprocessor(config)
    # decoder = PerceiverClassificationDecoder(
    #     config,
    #     num_channels=config.d_latents,
    #     trainable_position_encoding_kwargs=dict(num_channels=config.d_latents, index_dims=1),
    #     use_query_residual=True,
    # )
    # model = PerceiverModel(config, input_preprocessor=preprocessor, decoder=decoder)

    # # you can then do a forward pass as follows:
    # tokenizer = PerceiverTokenizer()
    # text = "hello world"
    # inputs = tokenizer(text, return_tensors="pt").input_ids

    # with torch.no_grad():
    #     outputs = model(inputs=inputs)
    #     logits = outputs.logits
    #     print(list(logits.shape))