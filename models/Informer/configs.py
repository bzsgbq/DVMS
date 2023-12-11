import torch

class Config(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

configs = Config(
    model = 'informer',  # model of experiment, options: [informer, informerstack]
    enc_in = 3,  # encoder input size
    dec_in = 3,  # decoder input size 
    c_out = 3,  # output size 
    seq_len = 5,  # input sequence length of Informer encoder
    label_len = 5,  # start token length of Informer decoder
    pred_len = 25,  # prediction sequence length
    factor = 5,  # probsparse attn factor
    d_model = 512,  # dimension of model
    n_heads = 8,  # num of heads
    e_layers = 2,  # num of encoder layers
    d_layers = 1,  # num of decoder layers
    s_layers = 2,  # num of stack encoder layers (informerstack中的e_layers)
    d_ff = 2048,  # dimension of fcn in model
    dropout = 0.05,  # dropout
    attn = 'prob',  # attention used in encoder, options:[prob, full]
    embed = 'timeF',  # time features encoding, options:[timeF, fixed, learned]
    freq = 's',  # freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h
    activation = 'gelu',  # activation
    output_attention = False,  # whether to output attention in encoder
    distil = True,  # whether to use distilling in encoder
    mix = True,  # whether to use mix attention in generative decoder
)