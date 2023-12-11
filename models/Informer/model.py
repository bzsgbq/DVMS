import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from models.Informer.utils.masking import TriangularCausalMask, ProbMask
    from models.Informer.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
    from models.Informer.decoder import Decoder, DecoderLayer
    from models.Informer.attn import FullAttention, ProbAttention, AttentionLayer
    from models.Informer.embed import DataEmbedding
except:
    try:
        from Informer.utils.masking import TriangularCausalMask, ProbMask
        from Informer.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
        from Informer.decoder import Decoder, DecoderLayer
        from Informer.attn import FullAttention, ProbAttention, AttentionLayer
        from Informer.embed import DataEmbedding
    except:
        from utils.masking import TriangularCausalMask, ProbMask
        from encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
        from decoder import Decoder, DecoderLayer
        from attn import FullAttention, ProbAttention, AttentionLayer
        from embed import DataEmbedding


class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True, mix=True,
                device=None):
        super(Informer, self).__init__()
        self.device = device if device is not None else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            attn_layers=[
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            conv_layers=[
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            layers=[
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        if x_dec is None:
            # decoder input
            x_dec = torch.zeros([x_enc.shape[0], self.pred_len, x_enc.shape[-1]]).float().to(self.device)
            x_dec = torch.cat([x_enc[:, :self.label_len, :], x_dec], dim=1).float().to(self.device)
            # NOTE: 原代码中, 上面这句是x_enc[:, -self.label_len:, :], 即使用了末尾的label_len个数据作为decoder的输入; 此处改为使用前label_len个数据作为decoder的输入

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]


class InformerStack(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=[3,2,1], d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                output_attention = False, distil=True, mix=True,
                device=None):
        super(InformerStack, self).__init__()
        self.device = device if device is not None else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder

        inp_lens = list(range(len(e_layers))) # [0,1,2,...] you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                    d_model, n_heads, mix=False),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el-1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in e_layers]
        self.encoder = EncoderStack(encoders, inp_lens)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        if x_dec is None:
            # decoder input
            x_dec = torch.zeros([x_enc.shape[0], self.pred_len, x_enc.shape[-1]]).float()
            x_dec = torch.cat([x_enc[:, :self.label_len, :], x_dec], dim=1).float().to(self.device)
            # NOTE: 原代码中, 上面这句是x_enc[:, -self.label_len:, :], 即使用了末尾的label_len个数据作为decoder的输入; 此处改为使用前label_len个数据作为decoder的输入
        
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]


def create_informer_model(cfg):
    model_dict = {
        'informer': Informer,
        'informerstack': InformerStack,
    }
    assert cfg.model in model_dict, f'{cfg.model} is a unrecognized model type'
    e_layers = cfg.e_layers if cfg.model=='informer' else cfg.s_layers
    model = model_dict[cfg.model](
        enc_in = cfg.enc_in,
        dec_in = cfg.dec_in,
        c_out = cfg.c_out,
        seq_len = cfg.seq_len,
        label_len = cfg.label_len,
        out_len = cfg.pred_len,
        factor = cfg.factor,
        d_model = cfg.d_model, 
        n_heads = cfg.n_heads,
        e_layers = e_layers, # cfg.e_layers,
        d_layers = cfg.d_layers,
        d_ff = cfg.d_ff,
        dropout = cfg.dropout,
        attn = cfg.attn,
        embed = cfg.embed,
        freq = cfg.freq,
        activation = cfg.activation, 
        output_attention = cfg.output_attention,
        distil = cfg.distil,
        mix = cfg.mix,
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    ).float()
    return model
    

if __name__ == '__main__':
    from configs import configs as cfg
    model = create_informer_model(cfg)
    model = model.to(model.device)
    x = torch.rand(32, cfg.seq_len, cfg.enc_in).to(model.device)
    out = model(x)
    print(out.shape)
    