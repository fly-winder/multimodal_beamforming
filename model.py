import math
from collections import deque
from collections import OrderedDict
import numpy as np
import torch 
from torch import nn
import torch.nn.functional as F
from torchvision import models



class ImageCNN(nn.Module):
    """ 
    Encoder network for image input list.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
    """

    def __init__(self, image_features=3, features=64, normalize=True):
        super().__init__()
        self.normalize = normalize
        self.feature = nn.Sequential(OrderedDict([
            ('layer', ImageCNN._block(image_features, features, name="img_enc")),
            ('layer1',ImageCNN._block(features, features, name="img_enc1")),
            ('layer2', ImageCNN._block(features, features * 2, name="img_enc2")),
            ('layer3', ImageCNN._block(features * 2, features * 4, name="img_enc3")),
            ('layer4', ImageCNN._block(features * 4, features * 8, name="img_enc4"))
        ]))

    def forward(self, inputs):
        c = 0
        for x in inputs:
            if self.normalize:
                x = normalize_imagenet(x)
            c += self.features(x)
        return c

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm3d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm3d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                    (
                        name + "conv3",
                        # nn.Conv2d(features, features*2, kernel_size=(1, 1), stride=(2, 2), bias=False),
                        nn.Conv3d(features, features, kernel_size=1, stride=(1,2,2), bias=False),
                    ),
                    (name + "norm3",
                     nn.BatchNorm3d(features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),

                ]
            )
        )

def normalize_imagenet(x):
    """ Normalize input images according to ImageNet standards.
    Args:
        x (tensor): input images
    """
    x = x.clone()
    x[:, 0] = (x[:, 0]/255.0 - 0.485) / 0.229
    x[:, 1] = (x[:, 1]/255.0 - 0.456) / 0.224
    x[:, 2] = (x[:, 2]/255.0 - 0.406) / 0.225
    return x


class LidarEncoder(nn.Module):
    """
    Encoder network for LiDAR input list
    Args:
        num_classes: output feature dimension
        in_channels: input channels
    """

    def __init__(self, lidar_features=1, features=64):
        super().__init__()

        self._model = nn.Sequential(OrderedDict([
            ('layer', LidarEncoder._block(lidar_features, features, name="lidar_enc")),
            ('layer1',LidarEncoder._block(features, features, name="lidar_enc1")),
            ('layer2', LidarEncoder._block(features, features * 2, name="lidar_enc2")),
            ('layer3', LidarEncoder._block(features * 2, features * 4, name="lidar_enc3")),
            ('layer4', LidarEncoder._block(features * 4, features * 8, name="lidar_enc4"))
        ]))
        # for param in self._model.parameters():
        #     param.requires_grad = False

    def forward(self, inputs):
        features = 0
        for lidar_data in inputs:
            lidar_feature = self._model(lidar_data)
            features += lidar_feature
        return features

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm3d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm3d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                    (
                        name + "conv3",
                        # nn.Conv2d(features, features*2, kernel_size=(1, 1), stride=(2, 2), bias=False),
                        nn.Conv3d(features, features, kernel_size=1, stride=(1,2,2), bias=False),
                    ),
                    (name + "norm3",
                     nn.BatchNorm3d(features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),

                ]
            )
        )

class SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(True), # changed from GELU
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        B, T, C = x.size()

        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, n_embd, n_head, block_exp, n_layer, 
                    vert_anchors, horz_anchors, seq_len, 
                    embd_pdrop, attn_pdrop, resid_pdrop, config):
        super().__init__()
        self.n_embd = n_embd
        self.seq_len = seq_len
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors
        self.config = config

        # positional embedding parameter (learnable), image + lidar
        self.pos_emb = nn.Parameter(torch.zeros(1, (self.config.n_views + 2) * config.seq_len * vert_anchors * horz_anchors+2, n_embd))

        self.drop = nn.Dropout(embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, 
                        block_exp, attn_pdrop, resid_pdrop)
                        for layer in range(n_layer)])
        
        # decoder head
        self.ln_f = nn.LayerNorm(n_embd)

        self.block_size = seq_len
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.BatchNorm2d)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # create the pytorch optimizer object
        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        return optim_groups

    def forward(self, image_tensor, lidar_tensor, radar_tensor, gps):
        """
        Args:
            image_tensor (tensor): B*4*seq_len, C, H, W
            lidar_tensor (tensor): B*seq_len, C, H, W
            gps (tensor): ego-gps
        """
        
        bz = lidar_tensor.shape[0]
        h, w = lidar_tensor.shape[3:5]


        # pad token embeddings along number of tokens dimension
        token_embeddings = torch.cat([image_tensor, lidar_tensor, radar_tensor], dim=2).permute(0,1,2,4,3).contiguous()
        # print(token_embeddings.shape)
        token_embeddings = token_embeddings.view(bz, -1, self.n_embd) # (B, an * T, C)
        token_embeddings = torch.cat([token_embeddings,gps], dim=1)
        # add (learnable) positional embedding and gps embedding for all tokens
        x = self.drop(self.pos_emb + token_embeddings )  # (B, an * T, C)
        x = self.blocks(x) # (B, an * T, C)
        x = self.ln_f(x) # (B, an * T, C)
        pos_tensor_out = x[:, (self.config.n_views + 2) * self.config.seq_len*self.vert_anchors * self.horz_anchors:, :]
        x = x[:,:(self.config.n_views + 2) * self.config.seq_len*self.vert_anchors * self.horz_anchors,:]


        x = x.view(bz, (self.config.n_views + 2) * self.config.seq_len , self.vert_anchors, self.horz_anchors, self.n_embd)
        x = x.permute(0,4,1,3,2).contiguous() # same as token_embeddings



        image_tensor_out = x[:, :, :self.config.n_views* self.config.seq_len, :, :].contiguous()
        lidar_tensor_out = x[:, :, self.config.n_views* self.config.seq_len:(self.config.n_views+1)*self.seq_len, :, :].contiguous()
        radar_tensor_out = x[:, :, (self.config.n_views+1)*self.seq_len:, :, :].contiguous()
        return image_tensor_out, lidar_tensor_out, radar_tensor_out, pos_tensor_out


class Encoder(nn.Module):
    """
    Multi-scale Fusion Transformer for image + LiDAR feature fusion
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.maxpool = nn.MaxPool3d(kernel_size=(1,3,3),stride=(1,2,2),padding=(0,1,1))
        self.avgpool = nn.AdaptiveAvgPool3d((5,self.config.vert_anchors, self.config.horz_anchors))
        self.avgpool11 = nn.AdaptiveAvgPool3d((5,1,1))
        self.image_encoder = ImageCNN(3,64, normalize=True)
        self.lidar_encoder = LidarEncoder(1, 64)
        if config.add_velocity:
            self.radar_encoder = LidarEncoder(2,64)
        else:
            self.radar_encoder = LidarEncoder(4,64)

        self.vel_emb1 = nn.Linear(2, 64)
        self.vel_emb2 = nn.Linear(64, 128)
        self.vel_emb3 = nn.Linear(128, 256)
        self.vel_emb4 = nn.Linear(256, 512)

        self.transformer1 = GPT(n_embd=64,
                            n_head=config.n_head, 
                            block_exp=config.block_exp, 
                            n_layer=config.n_layer, 
                            vert_anchors=config.vert_anchors, 
                            horz_anchors=config.horz_anchors, 
                            seq_len=config.seq_len, 
                            embd_pdrop=config.embd_pdrop, 
                            attn_pdrop=config.attn_pdrop, 
                            resid_pdrop=config.resid_pdrop,
                            config=config)
        self.transformer2 = GPT(n_embd=128,
                            n_head=config.n_head, 
                            block_exp=config.block_exp, 
                            n_layer=config.n_layer, 
                            vert_anchors=config.vert_anchors, 
                            horz_anchors=config.horz_anchors, 
                            seq_len=config.seq_len, 
                            embd_pdrop=config.embd_pdrop, 
                            attn_pdrop=config.attn_pdrop, 
                            resid_pdrop=config.resid_pdrop,
                            config=config)
        self.transformer3 = GPT(n_embd=256,
                            n_head=config.n_head, 
                            block_exp=config.block_exp, 
                            n_layer=config.n_layer, 
                            vert_anchors=config.vert_anchors, 
                            horz_anchors=config.horz_anchors, 
                            seq_len=config.seq_len, 
                            embd_pdrop=config.embd_pdrop, 
                            attn_pdrop=config.attn_pdrop, 
                            resid_pdrop=config.resid_pdrop,
                            config=config)
        self.transformer4 = GPT(n_embd=512,
                            n_head=config.n_head, 
                            block_exp=config.block_exp, 
                            n_layer=config.n_layer, 
                            vert_anchors=config.vert_anchors, 
                            horz_anchors=config.horz_anchors, 
                            seq_len=config.seq_len, 
                            embd_pdrop=config.embd_pdrop, 
                            attn_pdrop=config.attn_pdrop, 
                            resid_pdrop=config.resid_pdrop,
                            config=config)

        
    def forward(self, image_list, lidar_list, radar_list, gps):
        '''
        Image + LiDAR feature fusion using transformers
        Args:
            image_list (list): list of input images
            lidar_list (list): list of input LiDAR BEV
            gps (tensor): input gps
        '''

        # RGB standardization
        if self.image_encoder.normalize:
            image_list = [normalize_imagenet(image_input) for image_input in image_list]
        # bz:4、h:256、w:256、
        bz, _, h, w = lidar_list[0].shape
        img_channel = image_list[0].shape[1] # Image channel 3
        lidar_channel = lidar_list[0].shape[1] # Lidar channel 1
        radar_channel = radar_list[0].shape[1] # Radar channel 2


        self.config.n_views = len(image_list) // self.config.seq_len

        # (B,3,5,256,256)
        image_tensor = torch.stack(image_list, dim=2)
        lidar_tensor = torch.stack(lidar_list, dim=2)
        radar_tensor = torch.stack(radar_list, dim=2)




        # (B,64,5,128,128)
        image_features = self.image_encoder.feature.layer.img_encconv1(image_tensor)
        image_features = self.image_encoder.feature.layer.img_encnorm1(image_features)
        image_features = self.image_encoder.feature.layer.img_encrelu1(image_features)
        image_features = self.maxpool(image_features)

        lidar_features = self.lidar_encoder._model.layer.lidar_encconv1(lidar_tensor)
        lidar_features = self.lidar_encoder._model.layer.lidar_encnorm1(lidar_features)
        lidar_features = self.lidar_encoder._model.layer.lidar_encrelu1(lidar_features)
        lidar_features = self.maxpool(lidar_features)
        radar_features = self.radar_encoder._model.layer.lidar_encconv1(radar_tensor)
        radar_features = self.radar_encoder._model.layer.lidar_encnorm1(radar_features)
        radar_features = self.radar_encoder._model.layer.lidar_encrelu1(radar_features)
        radar_features = self.maxpool(radar_features)

       
        # ------------------------- The first layer feature extraction ----------------------------
        # (B,64,5,64,64)
        image_features = self.image_encoder.feature.layer1(image_features)
        lidar_features = self.lidar_encoder._model.layer1(lidar_features)
        radar_features = self.radar_encoder._model.layer1(radar_features)

        # (B,64,5,8,8)
        image_embd_layer1 = self.avgpool(image_features)
        lidar_embd_layer1 = self.avgpool(lidar_features)
        radar_embd_layer1 = self.avgpool(radar_features)
        gps_embd_layer1 = self.vel_emb1(gps)
        image_features_layer1 = image_embd_layer1
        lidar_features_layer1 = lidar_embd_layer1
        radar_features_layer1 = radar_embd_layer1
        gps_features_layer1 = gps_embd_layer1
        
        # image_features_layer1, lidar_features_layer1, radar_features_layer1, gps_features_layer1 = self.transformer1(image_embd_layer1, lidar_embd_layer1, radar_embd_layer1, gps_embd_layer1)
        image_features_layer1 = F.interpolate(image_features_layer1, scale_factor=[1,8,8], mode='trilinear')
        lidar_features_layer1 = F.interpolate(lidar_features_layer1, scale_factor=[1,8,8], mode='trilinear')
        radar_features_layer1 = F.interpolate(radar_features_layer1, scale_factor=[1,8,8], mode='trilinear')
        image_features = image_features + image_features_layer1
        lidar_features = lidar_features + lidar_features_layer1
        radar_features = radar_features + radar_features_layer1

        # ------------------------- The second layer feature extraction ----------------------------
        # (B,128,5,32,32)
        image_features = self.image_encoder.feature.layer2(image_features)
        lidar_features = self.lidar_encoder._model.layer2(lidar_features)
        radar_features = self.radar_encoder._model.layer2(radar_features)

        # (B,128,5,8,8)
        image_embd_layer2 = self.avgpool(image_features)
        lidar_embd_layer2 = self.avgpool(lidar_features)
        radar_embd_layer2 = self.avgpool(radar_features)
        gps_embd_layer2 = self.vel_emb2(gps_features_layer1)

        image_features_layer2, lidar_features_layer2, radar_features_layer2, gps_features_layer2 = self.transformer2(image_embd_layer2, lidar_embd_layer2, radar_embd_layer2, gps_embd_layer2)
        image_features_layer2 = F.interpolate(image_features_layer2, scale_factor=[1,4,4], mode='trilinear')
        lidar_features_layer2 = F.interpolate(lidar_features_layer2, scale_factor=[1,4,4], mode='trilinear')
        radar_features_layer2 = F.interpolate(radar_features_layer2, scale_factor=[1,4,4], mode='trilinear')
        image_features = image_features + image_features_layer2
        lidar_features = lidar_features + lidar_features_layer2
        radar_features = radar_features + radar_features_layer2

        # ------------------------- The third layer feature extraction ----------------------------
        image_features = self.image_encoder.feature.layer3(image_features)
        lidar_features = self.lidar_encoder._model.layer3(lidar_features)
        radar_features = self.radar_encoder._model.layer3(radar_features)


       
        image_embd_layer3 = self.avgpool(image_features)
        lidar_embd_layer3 = self.avgpool(lidar_features)
        radar_embd_layer3 = self.avgpool(radar_features)
        gps_embd_layer3 = self.vel_emb3(gps_features_layer2)

        image_features_layer3, lidar_features_layer3, radar_features_layer3, gps_features_layer3 = \
            image_embd_layer3, lidar_embd_layer3, radar_embd_layer3, gps_embd_layer3

        # image_features_layer3, lidar_features_layer3, radar_features_layer3, gps_features_layer3 = self.transformer3(image_embd_layer3, lidar_embd_layer3, radar_embd_layer3, gps_embd_layer3)
        
        image_features_layer3 = F.interpolate(image_features_layer3, scale_factor=[1,2,2], mode='trilinear')
        lidar_features_layer3 = F.interpolate(lidar_features_layer3, scale_factor=[1,2,2], mode='trilinear')
        radar_features_layer3 = F.interpolate(radar_features_layer3, scale_factor=[1,2,2], mode='trilinear')
        image_features = image_features + image_features_layer3
        lidar_features = lidar_features + lidar_features_layer3
        radar_features = radar_features + radar_features_layer3

        # ------------------------- The fourth layer feature extraction ----------------------------
        
        image_features = self.image_encoder.feature.layer4(image_features)
        lidar_features = self.lidar_encoder._model.layer4(lidar_features)
        radar_features = self.radar_encoder._model.layer4(radar_features)

        # (B,512,5,8,8)
        image_embd_layer4 = self.avgpool(image_features)
        lidar_embd_layer4 = self.avgpool(lidar_features)
        radar_embd_layer4 = self.avgpool(radar_features)
        gps_embd_layer4 = self.vel_emb4(gps_features_layer3)





        image_features_layer4, lidar_features_layer4, radar_features_layer4, gps_features_layer4 = self.transformer4(image_embd_layer4, lidar_embd_layer4, radar_embd_layer4, gps_embd_layer4)
        # （1，512，5，8，8）
        image_features = image_features + image_features_layer4
        lidar_features = lidar_features + lidar_features_layer4
        radar_features = radar_features + radar_features_layer4

        image_features = self.avgpool11(image_features)
        image_features = torch.flatten(image_features, 2)
        image_features = image_features.permute(0,2,1)
        lidar_features = self.avgpool11(lidar_features)
        lidar_features = torch.flatten(lidar_features, 2)
        lidar_features = lidar_features.permute(0, 2, 1)
        radar_features = self.avgpool11(radar_features)
        radar_features = torch.flatten(radar_features, 2)
        radar_features = radar_features.permute(0, 2, 1)
        gps_features = gps_features_layer4

        fused_features = torch.cat([image_features, lidar_features, radar_features, gps_features], dim=1)
        

        fused_features = torch.sum(fused_features, dim=1)

        return fused_features


class TransFuser(nn.Module):
    '''
    Transformer-based feature fusion followed by GRU-based waypoint prediction network and PID controller
    '''

    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.config = config
        self.pred_len = config.pred_len
        self.encoder = Encoder(config).to(self.device)

        self.join = nn.Sequential(
                            nn.Linear(512, 256),
                            nn.ReLU(inplace=True),
                            nn.Linear(256, 128),
                            nn.ReLU(inplace=True),
                            nn.Linear(128, 64),
                        ).to(self.device)
        # self.decoder = nn.GRUCell(input_size=2, hidden_size=64).to(self.device)
        # self.output = nn.Linear(64, 2).to(self.device)
        
    def forward(self, image_list, lidar_list, radar_list, gps):
        '''
        Predicts waypoint from geometric feature projections of image + LiDAR input
        Args:
            image_list (list): list of input images
            lidar_list (list): list of input LiDAR BEV
            target_point (tensor): goal location registered to ego-frame
            gps (tensor): input gps
        '''
        fused_features = self.encoder(image_list, lidar_list, radar_list, gps)
        z = self.join(fused_features)

       

        return z

# from torchsummary import summary
#
# MyNet = TransFuser(config, args.device)
#
# summary(MyNet,(1,2,4),1,"cuda")

