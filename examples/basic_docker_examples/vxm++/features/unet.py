import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Unet(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:
        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self,
                 inshape=None,
                 infeats=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            infeats: Number of input features.
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
            half_res: Skip the last decoder upsampling. Default is False.
        """

        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # cache some parameters
        self.half_res = half_res

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf

    def forward(self, x):

        # encoder forward pass
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            x = self.pooling[level](x)

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if not self.half_res or level < (self.nb_levels - 2):
                x = self.upsampling[level](x)
                x = torch.cat([x, x_history.pop()], dim=1)

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)

        return x
    
#def default_unet_features():
#    nb_features = [
#        [16, 32, 32, 32],             # encoder
##        [32, 32, 32, 32, 32, 16, 16]  # decoder
#    ]
#    return nb_features


def default_unet_features():
    nb_features = [
        [32, 48, 48, 64],             # encoder
        [64, 48, 48, 48, 48, 32, 64]  # decoder
    ]
    return nb_features



class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.activation = nn.ReLU()#nn.LeakyReLU(0.2)
        self.main2 = Conv(out_channels, out_channels, 1, stride,0)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.activation2 = nn.ReLU()#nn.LeakyReLU(0.2)


    def forward(self, x):
        out = self.activation(self.norm(self.main(x)))
        out = self.activation2(self.norm2(self.main2(out)))
        return out


def weighted_sample(seg_fix,N,H,W,D):
    with torch.no_grad():
        weight_s = 1/(torch.bincount(seg_fix.long().reshape(-1))).float().pow(.5)
        weight_s[torch.isinf(weight_s)] = 0
        weight_s[torch.isnan(weight_s)] = 0
        #weight_s[weight_s==1] = 0
        weight_s[0] *= 10
        weight_s /= weight_s.mean()
        mask = F.max_pool3d(F.max_pool3d((seg_fix.view(1,1,H,W,D).cuda()>0).float(),5,stride=1,padding=2),5,stride=1,padding=2)
        indx = mask.view(-1).nonzero()
        w_idx = weight_s[seg_fix.reshape(-1)[indx]].squeeze()
        w_idx[torch.isinf(w_idx)] = 0
        w_idx[torch.isnan(w_idx)] = 0
        w_idx[w_idx<0] = 0
        ident = F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda(),(1,1,H,W,D))
        randweight = torch.utils.data.WeightedRandomSampler(w_idx,N,replacement=False)
        dataset = torch.utils.data.TensorDataset(torch.arange(len(w_idx)))
        loader = torch.utils.data.DataLoader(dataset, batch_size=N, sampler=randweight)
        for _, sample_batched in enumerate(loader):
            indices = sample_batched[0]
        #
        sample_index = indx[indices]
#        sample_label = seg_fix.reshape(-1)[sample_index]
        sample_xyz = ident.view(-1,3)[sample_index.view(-1),:]
        
    return sample_xyz
def create_unet(infeats,inshape=(128,128,128)):


    nb_unet_features=None
    nb_unet_levels=None
    unet_feat_mult=1
    nb_unet_conv_per_level=1
    int_steps=7
    int_downsize=2
    bidir=False
    use_probs=False
    src_feats=1
    trg_feats=1
    unet_half_res=True

    unet_model = Unet(inshape,infeats,nb_features=nb_unet_features,nb_levels=nb_unet_levels,\
                feat_mult=unet_feat_mult,nb_conv_per_level=nb_unet_conv_per_level,half_res=unet_half_res,)
    return unet_model

