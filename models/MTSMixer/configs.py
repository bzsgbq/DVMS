class Config(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

configs = Config(
    seq_len = 5,
    pred_len = 25,
    enc_in = 3,
    d_model = 512,
    e_layers = 6,
    d_ff = 2048,

    individual = False,
    rev = False,
    fac_T = False,
    sampling = 2,  # 只有在fac_T为True时才会用到;
    fac_C = True,
    norm = False,
    # refine = False,  # 原代码中把关于refine的部分注释掉了;
)


# import argparse
# parser = argparse.ArgumentParser(description='Multivariate Time Series Forecasting')

# # forecasting task
# parser.add_argument('--seq_len', type=int, default=5, help='input sequence length')
# # parser.add_argument('--label_len', type=int, default=5, help='start token length')
# parser.add_argument('--pred_len', type=int, default=25, help='prediction sequence length')

# parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')
# # parser.add_argument('--seg', type=int, default=20, help='prediction plot segments')
# parser.add_argument('--rev', action='store_true', default=False, help='whether to apply RevIN')
# parser.add_argument('--norm', action='store_false', default=True, help='whether to apply LayerNorm')
# parser.add_argument('--fac_T', action='store_true', default=False, help='whether to apply factorized temporal interaction')
# parser.add_argument('--sampling', type=int, default=2, help='the number of downsampling in factorized temporal interaction')
# parser.add_argument('--fac_C', action='store_true', default=False, help='whether to apply factorized channel interaction')
# parser.add_argument('--refine', action='store_true', default=False, help='whether to refine the linear prediction')
# # parser.add_argument('--mat', type=int, default=0, help='option: [0-random, 1-identity]')

# # model 
# # parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + positional embedding')
# parser.add_argument('--enc_in', type=int, default=3, help='encoder input size')
# # parser.add_argument('--dec_in', type=int, default=3, help='decoder input size')
# # parser.add_argument('--c_out', type=int, default=3, help='output size')
# parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
# # parser.add_argument('--n_heads', type=int, default=1, help='num of heads')
# parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
# # parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
# parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
# # parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
# # parser.add_argument('--factor', type=int, default=1, help='attn factor')
# # parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
# # parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
# # parser.add_argument('--activation', type=str, default='gelu', help='activation')
# # parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
# # parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# args = parser.parse_args()