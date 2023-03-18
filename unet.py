import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed 
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


# class Dense(nn.Module):
#   """A fully connected layer that reshapes outputs to feature maps."""
#   def __init__(self, input_dim, output_dim):
#     super().__init__()
#     self.dense = nn.Linear(input_dim, output_dim)
#   def forward(self, x):
#     return self.dense(x)[..., None, None]


# class UNet(nn.Module):
#   def __init__(self, ranks, channels=[32, 64, 128, 256], embed_dim=256):

#     super().__init__()

#     # Gaussian random feature embedding layer for ranks
#     self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim, ranks=ranks),
#          nn.Linear(embed_dim, embed_dim))
    
#     # Encoding layers where the resolution decreases
#     self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1)
#     self.dense1 = Dense(embed_dim, channels[0])
#     self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
#     self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2)
#     self.dense2 = Dense(embed_dim, channels[1])
#     self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
#     self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2)
#     self.dense3 = Dense(embed_dim, channels[2])
#     self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
#     self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2)
#     self.dense4 = Dense(embed_dim, channels[3])
#     self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])    

#     # # Decoding layers for left singular vector
#     # self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False)
#     # self.dense5 = Dense(embed_dim, channels[2])
#     # self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
#     # self.tconv3 = nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 3, stride=2, bias=False, output_padding=1)    
#     # self.dense6 = Dense(embed_dim, channels[1])
#     # self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
#     # self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 3, stride=2, bias=False, output_padding=1)    
#     # self.linear = nn.Linear(21632, 28)

#     # # Decoding layers for right singular vector
#     # self.tconv4_r = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False)
#     # self.dense5_r = Dense(embed_dim, channels[2])
#     # self.tgnorm4_r = nn.GroupNorm(32, num_channels=channels[2])
#     # self.tconv3_r = nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 3, stride=2, bias=False, output_padding=1)
#     # self.dense6_r = Dense(embed_dim, channels[1])
#     # self.tgnorm3_r = nn.GroupNorm(32, num_channels=channels[1])
#     # self.tconv2_r = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 3, stride=2, bias=False, output_padding=1)
#     # self.linear_r = nn.Linear(21632, 28)

#     # self.linear_sig = nn.Linear(1024, 1)
#     # Decoding layers where the resolution increases
#     self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2)
#     self.dense5 = Dense(embed_dim, channels[2])
#     self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
#     self.tconv3 = nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 3, stride=2, output_padding=1)    
#     self.dense6 = Dense(embed_dim, channels[1])
#     self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
#     self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 3, stride=2, output_padding=1)    
#     self.dense7 = Dense(embed_dim, channels[0])
#     self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
#     self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 1, 3, stride=1)
    
#     # The swish activation function
#     self.act = nn.ReLU()#lambda x: x * torch.sigmoid(x)
  
#   def forward(self, x, t): 
#     # Obtain the Gaussian random feature embedding for t   
#     #embed = self.act(self.embed(t))    
#     # Encoding path
#     h1 = self.conv1(x)    
#     ## Incorporate information from rank embedding
#     #h1 += self.dense1(embed)
#     ## Group normalization
#     #h1 = self.gnorm1(h1)
#     h1 = self.act(h1)
#     h2 = self.conv2(h1)
#     #h2 += self.dense2(embed)
#     #h2 = self.gnorm2(h2)
#     h2 = self.act(h2)
#     h3 = self.conv3(h2)
#     #h3 += self.dense3(embed)
#     #h3 = self.gnorm3(h3)
#     h3 = self.act(h3)
#     h4 = self.conv4(h3)
#     #h4 += self.dense4(embed)
#     #h4 = self.gnorm4(h4)
#     h4 = self.act(h4)

    

#     # sig = self.linear_sig(h4.view(h4.shape[0], -1))

#     # # Decoding path
#     # h = self.tconv4(h4)
#     # ## Skip connection from the encoding path
#     # #h += self.dense5(embed)
#     # h = self.tgnorm4(h)
#     # h = self.act(h)
#     # h = self.tconv3(torch.cat([h, h3], dim=1))
#     # #h += self.dense6(embed)
#     # h = self.tgnorm3(h)
#     # h = self.act(h)
#     # h = self.tconv2(torch.cat([h, h2], dim=1))
#     # h = self.act(h)
#     # h = self.linear(h.view(h.shape[0], -1))

#     # # Decoding path for right singular vector
#     # h_r = self.tconv4_r(h4)
#     # ## Skip connection from the encoding path
#     # #h_r += self.dense5_r(embed)
#     # h_r = self.tgnorm4_r(h_r)
#     # h_r = self.act(h_r)
#     # h_r = self.tconv3_r(torch.cat([h_r, h3], dim=1))
#     # #h_r += self.dense6_r(embed)
#     # h_r = self.tgnorm3_r(h_r)
#     # h_r = self.act(h_r)
#     # h_r = self.tconv2_r(torch.cat([h_r, h2], dim=1))
#     # h_r = self.act(h_r)
#     # h_r = self.linear_r(h_r.view(h_r.shape[0], -1))
#         # Decoding path
#     h = self.tconv4(h4)
#     ## Skip connection from the encoding path
#     #h += self.dense5(embed)
#     #h = self.tgnorm4(h)
#     h = self.act(h)
#     h = self.tconv3(torch.cat([h, h3], dim=1))
#     #h += self.dense6(embed)
#     #h = self.tgnorm3(h)
#     h = self.act(h)
#     h = self.tconv2(torch.cat([h, h2], dim=1))
#     #h += self.dense7(embed)
#     #h = self.tgnorm2(h)
#     h = self.act(h)
#     h = self.tconv1(torch.cat([h, h1], dim=1))

#     return h
  
# class MLP(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim1,hidden_dim2, output_dim):
#         super(MLP, self).__init__()
#         self.fc1 = torch.nn.Linear(input_dim, hidden_dim1)
#         self.fc2 = torch.nn.Linear(hidden_dim1, hidden_dim2)
#         self.fc3 = torch.nn.Linear(hidden_dim2, output_dim)
#         self.relu = torch.nn.ReLU()
        
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.fc3(x)
       
#         return x
    



# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from torch.nn.modules.normalization import GroupNorm


# def get_norm(norm, num_channels, num_groups):
#     if norm == "in":
#         return nn.InstanceNorm2d(num_channels, affine=True)
#     elif norm == "bn":
#         return nn.BatchNorm2d(num_channels)
#     elif norm == "gn":
#         return nn.GroupNorm(num_groups, num_channels)
#     elif norm is None:
#         return nn.Identity()
#     else:
#         raise ValueError("unknown normalization type")


# class PositionalEmbedding(nn.Module):
#     __doc__ = r"""Computes a positional embedding of timesteps.
#     Input:
#         x: tensor of shape (N)
#     Output:
#         tensor of shape (N, dim)
#     Args:
#         dim (int): embedding dimension
#         scale (float): linear scale to be applied to timesteps. Default: 1.0
#     """

#     def __init__(self, dim, scale=1.0):
#         super().__init__()
#         assert dim % 2 == 0
#         self.dim = dim
#         self.scale = scale

#     def forward(self, x):
#         device = x.device
#         half_dim = self.dim // 2
#         emb = math.log(10000) / half_dim
#         emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
#         emb = torch.outer(x * self.scale, emb)
#         emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
#         return emb


# class Downsample(nn.Module):
#     __doc__ = r"""Downsamples a given tensor by a factor of 2. Uses strided convolution. Assumes even height and width.
#     Input:
#         x: tensor of shape (N, in_channels, H, W)
#         time_emb: ignored
#         y: ignored
#     Output:
#         tensor of shape (N, in_channels, H // 2, W // 2)
#     Args:
#         in_channels (int): number of input channels
#     """

#     def __init__(self, in_channels):
#         super().__init__()

#         self.downsample = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)
    
#     def forward(self, x, time_emb, y):
#         if x.shape[2] % 2 == 1:
#             raise ValueError("downsampling tensor height should be even")
#         if x.shape[3] % 2 == 1:
#             raise ValueError("downsampling tensor width should be even")

#         return self.downsample(x)


# class Upsample(nn.Module):
#     __doc__ = r"""Upsamples a given tensor by a factor of 2. Uses resize convolution to avoid checkerboard artifacts.
#     Input:
#         x: tensor of shape (N, in_channels, H, W)
#         time_emb: ignored
#         y: ignored
#     Output:
#         tensor of shape (N, in_channels, H * 2, W * 2)
#     Args:
#         in_channels (int): number of input channels
#     """

#     def __init__(self, in_channels):
#         super().__init__()

#         self.upsample = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode="nearest"),
#             nn.Conv2d(in_channels, in_channels, 3, padding=1),
#         )
    
#     def forward(self, x, time_emb, y):
#         return self.upsample(x)


# class AttentionBlock(nn.Module):
#     __doc__ = r"""Applies QKV self-attention with a residual connection.
    
#     Input:
#         x: tensor of shape (N, in_channels, H, W)
#         norm (string or None): which normalization to use (instance, group, batch, or none). Default: "gn"
#         num_groups (int): number of groups used in group normalization. Default: 32
#     Output:
#         tensor of shape (N, in_channels, H, W)
#     Args:
#         in_channels (int): number of input channels
#     """
#     def __init__(self, in_channels, norm="gn", num_groups=32):
#         super().__init__()
        
#         self.in_channels = in_channels
#         self.norm = get_norm(norm, in_channels, num_groups)
#         self.to_qkv = nn.Conv2d(in_channels, in_channels * 3, 1)
#         self.to_out = nn.Conv2d(in_channels, in_channels, 1)

#     def forward(self, x):
#         b, c, h, w = x.shape
#         q, k, v = torch.split(self.to_qkv(self.norm(x)), self.in_channels, dim=1)

#         q = q.permute(0, 2, 3, 1).view(b, h * w, c)
#         k = k.view(b, c, h * w)
#         v = v.permute(0, 2, 3, 1).view(b, h * w, c)

#         dot_products = torch.bmm(q, k) * (c ** (-0.5))
#         assert dot_products.shape == (b, h * w, h * w)

#         attention = torch.softmax(dot_products, dim=-1)
#         out = torch.bmm(attention, v)
#         assert out.shape == (b, h * w, c)
#         out = out.view(b, h, w, c).permute(0, 3, 1, 2)

#         return self.to_out(out) + x


# class ResidualBlock(nn.Module):
#     __doc__ = r"""Applies two conv blocks with resudual connection. Adds time and class conditioning by adding bias after first convolution.
#     Input:
#         x: tensor of shape (N, in_channels, H, W)
#         time_emb: time embedding tensor of shape (N, time_emb_dim) or None if the block doesn't use time conditioning
#         y: classes tensor of shape (N) or None if the block doesn't use class conditioning
#     Output:
#         tensor of shape (N, out_channels, H, W)
#     Args:
#         in_channels (int): number of input channels
#         out_channels (int): number of output channels
#         time_emb_dim (int or None): time embedding dimension or None if the block doesn't use time conditioning. Default: None
#         num_classes (int or None): number of classes or None if the block doesn't use class conditioning. Default: None
#         activation (function): activation function. Default: torch.nn.functional.relu
#         norm (string or None): which normalization to use (instance, group, batch, or none). Default: "gn"
#         num_groups (int): number of groups used in group normalization. Default: 32
#         use_attention (bool): if True applies AttentionBlock to the output. Default: False
#     """

#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         dropout,
#         time_emb_dim=None,
#         num_classes=None,
#         activation=F.relu,
#         norm="gn",
#         num_groups=32,
#         use_attention=False,
#     ):
#         super().__init__()

#         self.activation = activation

#         self.norm_1 = get_norm(norm, in_channels, num_groups)
#         self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

#         self.norm_2 = get_norm(norm, out_channels, num_groups)
#         self.conv_2 = nn.Sequential(
#             nn.Dropout(p=dropout),
#             nn.Conv2d(out_channels, out_channels, 3, padding=1),
#         )

#         self.time_bias = nn.Linear(time_emb_dim, out_channels) if time_emb_dim is not None else None
#         self.class_bias = nn.Embedding(num_classes, out_channels) if num_classes is not None else None

#         self.residual_connection = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
#         self.attention = nn.Identity() if not use_attention else AttentionBlock(out_channels, norm, num_groups)
    
#     def forward(self, x, time_emb=None, y=None):
#         out = self.activation(self.norm_1(x))
        
#         out = self.conv_1(out)

#         if self.time_bias is not None:
#             if time_emb is None:
#                 raise ValueError("time conditioning was specified but time_emb is not passed")
#             out += self.time_bias(self.activation(time_emb))[:, :, None, None]

#         if self.class_bias is not None:
#             if y is None:
#                 raise ValueError("class conditioning was specified but y is not passed")

#             out += self.class_bias(y)[:, :, None, None]

#         out = self.activation(self.norm_2(out))
#         out = self.conv_2(out) + self.residual_connection(x)
#         out = self.attention(out)

#         return out


# class UNet2(nn.Module):
#     __doc__ = """UNet model used to estimate noise.
#     Input:
#         x: tensor of shape (N, in_channels, H, W)
#         time_emb: time embedding tensor of shape (N, time_emb_dim) or None if the block doesn't use time conditioning
#         y: classes tensor of shape (N) or None if the block doesn't use class conditioning
#     Output:
#         tensor of shape (N, out_channels, H, W)
#     Args:
#         img_channels (int): number of image channels
#         base_channels (int): number of base channels (after first convolution)
#         channel_mults (tuple): tuple of channel multiplers. Default: (1, 2, 4, 8)
#         time_emb_dim (int or None): time embedding dimension or None if the block doesn't use time conditioning. Default: None
#         time_emb_scale (float): linear scale to be applied to timesteps. Default: 1.0
#         num_classes (int or None): number of classes or None if the block doesn't use class conditioning. Default: None
#         activation (function): activation function. Default: torch.nn.functional.relu
#         dropout (float): dropout rate at the end of each residual block
#         attention_resolutions (tuple): list of relative resolutions at which to apply attention. Default: ()
#         norm (string or None): which normalization to use (instance, group, batch, or none). Default: "gn"
#         num_groups (int): number of groups used in group normalization. Default: 32
#         initial_pad (int): initial padding applied to image. Should be used if height or width is not a power of 2. Default: 0
#     """

#     def __init__(
#         self,
#         img_channels,
#         base_channels,
#         channel_mults=(1, 2, 4, 8),
#         num_res_blocks=2,
#         time_emb_dim=None,
#         time_emb_scale=1.0,
#         num_classes=None,
#         activation=F.relu,
#         dropout=0.1,
#         attention_resolutions=(),
#         norm="gn",
#         num_groups=32,
#         initial_pad=0,
#     ):
#         super().__init__()

#         self.activation = activation
#         self.initial_pad = initial_pad

#         self.num_classes = num_classes
#         self.time_mlp = nn.Sequential(
#             PositionalEmbedding(base_channels, time_emb_scale),
#             nn.Linear(base_channels, time_emb_dim),
#             nn.SiLU(),
#             nn.Linear(time_emb_dim, time_emb_dim),
#         ) if time_emb_dim is not None else None
    
#         self.init_conv = nn.Conv2d(img_channels, base_channels, 3, padding=1)

#         self.downs = nn.ModuleList()
#         self.ups = nn.ModuleList()

#         channels = [base_channels]
#         now_channels = base_channels

#         for i, mult in enumerate(channel_mults):
#             out_channels = base_channels * mult

#             for _ in range(num_res_blocks):
#                 self.downs.append(ResidualBlock(
#                     now_channels,
#                     out_channels,
#                     dropout,
#                     time_emb_dim=time_emb_dim,
#                     num_classes=num_classes,
#                     activation=activation,
#                     norm=norm,
#                     num_groups=num_groups,
#                     use_attention=i in attention_resolutions,
#                 ))
#                 now_channels = out_channels
#                 channels.append(now_channels)
            
#             if i != len(channel_mults) - 1:
#                 self.downs.append(Downsample(now_channels))
#                 channels.append(now_channels)
        

#         self.mid = nn.ModuleList([
#             ResidualBlock(
#                 now_channels,
#                 now_channels,
#                 dropout,
#                 time_emb_dim=time_emb_dim,
#                 num_classes=num_classes,
#                 activation=activation,
#                 norm=norm,
#                 num_groups=num_groups,
#                 use_attention=True,
#             ),
#             ResidualBlock(
#                 now_channels,
#                 now_channels,
#                 dropout,
#                 time_emb_dim=time_emb_dim,
#                 num_classes=num_classes,
#                 activation=activation,
#                 norm=norm,
#                 num_groups=num_groups,
#                 use_attention=False,
#             ),
#         ])

#         for i, mult in reversed(list(enumerate(channel_mults))):
#             out_channels = base_channels * mult

#             for _ in range(num_res_blocks + 1):
#                 self.ups.append(ResidualBlock(
#                     channels.pop() + now_channels,
#                     out_channels,
#                     dropout,
#                     time_emb_dim=time_emb_dim,
#                     num_classes=num_classes,
#                     activation=activation,
#                     norm=norm,
#                     num_groups=num_groups,
#                     use_attention=i in attention_resolutions,
#                 ))
#                 now_channels = out_channels
            
#             if i != 0:
#                 self.ups.append(Upsample(now_channels))
        
#         assert len(channels) == 0
        
#         self.out_norm = get_norm(norm, base_channels, num_groups)
#         self.out_conv = nn.Conv2d(base_channels, img_channels, 3, padding=1)
    
#     def forward(self, x, time=None, y=None):
#         ip = self.initial_pad
#         if ip != 0:
#             x = F.pad(x, (ip,) * 4)

#         if self.time_mlp is not None:
#             if time is None:
#                 raise ValueError("time conditioning was specified but tim is not passed")
            
#             time_emb = self.time_mlp(time)
#         else:
#             time_emb = None
        
#         if self.num_classes is not None and y is None:
#             raise ValueError("class conditioning was specified but y is not passed")
        
#         x = self.init_conv(x)

#         skips = [x]

#         for layer in self.downs:
#             x = layer(x, time_emb, y)
#             skips.append(x)
        
#         for layer in self.mid:
#             x = layer(x, time_emb, y)
        
#         for layer in self.ups:
#             if isinstance(layer, ResidualBlock):
#                 x = torch.cat([x, skips.pop()], dim=1)
#             x = layer(x, time_emb, y)

#         x = self.activation(self.out_norm(x))
#         x = self.out_conv(x)
        
#         if self.initial_pad != 0:
#             return x[:, :, ip:-ip, ip:-ip]
#         else:
#             return x
  
class MyBlock(nn.Module):
    def __init__(self, shape, out_c, block_layer):
        super(MyBlock, self).__init__()
        self.w1 = nn.Linear(shape, out_c)
        self.block_layer = block_layer
        for nl in range(2,self.block_layer+1):
            setattr(self,"w_"+str(nl),nn.Linear(out_c,out_c))
        self.activation = lambda x: x * torch.sigmoid(x)


    def forward(self, x):
        out = self.w1(x)
        out = self.activation(out)
        for nl in range(2,self.block_layer+1):
            out = self.activation(getattr(self,"w_"+str(nl))(out))
        return out
  

class UNet_MLP(nn.Module):
    def __init__(self, input_dim, time_emb_dim, num_classes = None, class_emb_dim = 512, ranks = 28, scale = 9, block_layer = 2):
        super(UNet_MLP, self).__init__()

        self.ranks = ranks
        # Sinusoidal embedding
        self.act = lambda x: x * torch.sigmoid(x)
        # Sinusoidal embedding
        self.time_embed = nn.Sequential(GaussianFourierProjection(embed_dim=time_emb_dim),
         nn.Linear(time_emb_dim, time_emb_dim))
        self.class_bias = nn.Embedding(num_classes, class_emb_dim) if num_classes is not None else None
        
        self.input_dim = input_dim
        
        # First half
        first_num = 2 ** scale
        self.x_embed = MyBlock(input_dim * 2,first_num,block_layer)
        
        self.te1 = self._make_emb(time_emb_dim, first_num)
        if num_classes is not None:
            self.ce1 = self._make_emb(class_emb_dim, first_num)
        self.b1 = MyBlock(first_num, first_num,block_layer)

        

        second_num = first_num // 2
        self.down1 = MyBlock(first_num,second_num,block_layer)
        
        self.te2 = self._make_emb(time_emb_dim, second_num)
        if num_classes is not None:
            self.ce2 = self._make_emb(class_emb_dim, second_num)
        self.b2 = MyBlock(second_num,second_num,block_layer)
    
        
        third_num = second_num // 2
        self.down2 = MyBlock(second_num,third_num,block_layer)


        # Bottleneck
        self.te_mid = self._make_emb(time_emb_dim, third_num)
        if num_classes is not None:
            self.ce_mid = self._make_emb(class_emb_dim, third_num)
        self.b_mid = MyBlock(third_num, third_num,block_layer)
    

        # Second half
        self.up1 = MyBlock(third_num, second_num,block_layer)
        self.te3 = self._make_emb(time_emb_dim, first_num)
        if num_classes is not None:
            self.ce3 = self._make_emb(class_emb_dim, first_num)
        self.b3 = MyBlock(first_num, second_num,block_layer)

        self.up2 = MyBlock(second_num, first_num,block_layer)
        self.te4 = self._make_emb(time_emb_dim, first_num * 2)
        if num_classes is not None:
            self.ce4 = self._make_emb(class_emb_dim, first_num * 2)
        self.b4 = MyBlock(first_num * 2, first_num,block_layer)
        

        self.final = nn.Linear(first_num, input_dim)

    def forward(self, x0, ti, cls=None):

        t = self.act(self.time_embed(ti / self.ranks))
        y = self.class_bias(cls) if self.class_bias is not None else None
        x = self.x_embed(torch.cat((x0, x0), dim=1))
        
        out1 = x + self.te1(t)
        if self.class_bias is not None:
            out1 = out1 + self.ce1(y)
        out1 = self.b1(out1)   # (N, first_num) 
        out2 = self.down1(out1) + self.te2(t)   # (N, second_num)
        if self.class_bias is not None:
            out2 = out2 + self.ce2(y)
        out2 = self.b2(out2)    # (N, second_num)
        out_mid = self.down2(out2)+ self.te_mid(t)   # (N, third_num)
        if self.class_bias is not None:
            out_mid = out_mid + self.ce_mid(y)
        out_mid = self.b_mid(out_mid)   # (N, third_num)

        out3 = torch.cat((out2, self.up1(out_mid)), dim=1)  # (N, first_num)
        out4 = out3+ self.te3(t)
        if self.class_bias is not None:
            out4 = out4 + self.ce3(y)
        out4 = self.b3(out4)    # (N, second)

        out5 = torch.cat((out1, self.up2(out4)), dim=1)  # (N, first_num * 2)
        out6 = out5+ self.te4(t)
        if self.class_bias is not None:
            out6 = out6 + self.ce4(y)
        out6 = self.b4(out6)    # (N, first_num)

        out = self.final(out6) # (N, out)
        return out 

    def _make_emb(self, dim_in, dim_out):
        return nn.Linear(dim_in, dim_out)