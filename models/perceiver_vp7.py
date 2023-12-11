from transformers.models.perceiver.modeling_perceiver import *
from transformers.models.perceiver.configuration_perceiver import PerceiverConfig

try:
    from models.models_utils import get_model_size, get_xyz_grid, salmap2posalfeat
    from models.perceiver_vp import PerceiverVPPreprocessor, PerceiverVPDecoder, NUM_BANDS
except:
    from models_utils import get_model_size, get_xyz_grid, salmap2posalfeat
    from perceiver_vp import PerceiverVPPreprocessor, PerceiverVPDecoder, NUM_BANDS


'''
version 7: 在version6的基础上, 我们分别使用一个线性层, 将pos和sal的维度映射到HIDDEN_SIZE, 
然后再输入到perceiver中.

TODO: transformers.models.perceiver.modeling_perceiver.PerceiverModel.post_init会触发警告:
/usr/local/anaconda3/envs/gbq_pytorch/lib/python3.8/site-packages/torch/nn/modules/container.py:587: UserWarning: Setting attributes on ParameterDict is not supported.
  warnings.warn("Setting attributes on ParameterDict is not supported.")
'''


CAT_OR_ADD = "concat"  # "concat" or "add"
assert CAT_OR_ADD in ["concat", "add"]
HIDDEN_SIZE = 8*NUM_BANDS-1 if CAT_OR_ADD == "concat" else 2*NUM_BANDS+1

PerceiverPosPreprocessor = PerceiverVPPreprocessor
PerceiverSalPreprocessor = PerceiverVPPreprocessor


class PerceiverForViewpointPrediction7(PerceiverPreTrainedModel):
    def __init__(self, config, m_window, h_window):
        super().__init__(config)
        self.m_window = m_window
        self.h_window = h_window

        self.perceiver = PerceiverModel(
            config,
            input_preprocessor=PerceiverMultimodalPreprocessor(
                min_padding_size=7 if CAT_OR_ADD == "add" else 0,
                modalities={
                    "pos": PerceiverPosPreprocessor(
                        config,
                        m_window=self.m_window,
                        h_window=self.h_window,
                        feat_dim=HIDDEN_SIZE,
                        position_encoding_type="fourier",
                        concat_or_add_pos=CAT_OR_ADD,
                    ),
                    "sal": PerceiverSalPreprocessor(
                        config,
                        m_window=self.m_window,
                        h_window=self.h_window,
                        feat_dim=HIDDEN_SIZE,
                        position_encoding_type="fourier",
                        concat_or_add_pos=CAT_OR_ADD,
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
    


class PerceiverVPModel7(nn.Module):
    def __init__(self, config, M_WINDOW, H_WINDOW, salmap_shape):
        super(PerceiverVPModel7, self).__init__()
        self.m_window = M_WINDOW
        self.h_window = H_WINDOW
        num_data = salmap_shape[0]  # salmap_shap: (N, 4)

        self.fc_pos = nn.Linear(3, HIDDEN_SIZE)
        self.fc_sal = nn.Linear(4*num_data, HIDDEN_SIZE)
        self.perceiver = PerceiverForViewpointPrediction7(config, self.m_window, self.h_window)

    def forward(self, x):
        enc_pos_in, m_sal_in, h_sal_in = x
        enc_pos_in = enc_pos_in[:, :, :3]# if self.pos_dim == 3 else enc_pos_in[:, :, -2:]
        enc_sal_in = torch.cat([m_sal_in, h_sal_in], dim=1).flatten(start_dim=2)
        enc_pos_in = self.fc_pos(enc_pos_in)
        enc_sal_in = self.fc_sal(enc_sal_in)
        inputs = dict(pos=enc_pos_in, sal=enc_sal_in)
        outputs = self.perceiver(inputs)
        return outputs.logits


    def predict(self, x):
        return self.forward(x)[:, -self.h_window:, :]


if __name__ == "__main__":
    cfg = PerceiverConfig(
        num_latents=64,
        d_latents=512,
        num_blocks=4,
        num_self_attends_per_block=4,
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
    salmap_shape = (100, 4)
    
    model = PerceiverVPModel7(cfg, m_window, h_window, salmap_shape)
    pos = torch.ones(batch_size, m_window, 3)
    m_sal = 2 * torch.ones(batch_size, m_window, salmap_shape[0], salmap_shape[1])
    h_sal = 2 * torch.ones(batch_size, h_window, salmap_shape[0], salmap_shape[1])
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