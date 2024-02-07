import torch
import torch.nn as nn
import torch.distributions as td
import numpy as np

def conv_out(h_in, padding, kernel_size, stride):
    return int((h_in + 2. * padding - (kernel_size - 1.) - 1.) / stride + 1.)

def output_padding(h_in, conv_out, padding, kernel_size, stride):
    return h_in - (conv_out - 1) * stride + 2 * padding - (kernel_size - 1) - 1

def conv_out_shape(h_in, padding, kernel_size, stride):
    return tuple(conv_out(x, padding, kernel_size, stride) for x in h_in)

def output_padding_shape(h_in, conv_out, padding, kernel_size, stride):
    return tuple(output_padding(h_in[i], conv_out[i], padding, kernel_size, stride) for i in range(len(h_in)))


class ObsEncoder(nn.Module):
    def __init__(self, in_shape, out_shape, game_type = 'atari', device='cuda'):
        super(ObsEncoder, self).__init__()
        
        self.activation = nn.ELU
        self.shape = in_shape
        self.out_shape = out_shape
        self.game_type = game_type
        if game_type == 'minatari':
            self.depth = 16
            self.kernel = 3
            
            self.conv1 = nn.Sequential(nn.Conv2d(in_shape[0], self.depth, self.kernel), self.activation())
            self.conv2 = nn.Sequential(nn.Conv2d(self.depth, self.depth*2, self.kernel), self.activation())
            self.conv3 = nn.Sequential(nn.Conv2d(self.depth*2, self.depth*4, self.kernel), self.activation())

        else:
            self.depth = 32
            self.kernel = 4

            self.conv1 = nn.Sequential(nn.Conv2d(in_shape[0], self.depth, self.kernel, stride=2), self.activation())
            self.conv2 = nn.Sequential(nn.Conv2d(self.depth, self.depth*2, self.kernel, stride=2), self.activation())
            self.conv3 = nn.Sequential(nn.Conv2d(self.depth*2, self.depth*4, self.kernel, stride=2), self.activation())
            self.conv4 = nn.Sequential(nn.Conv2d(self.depth*4, self.depth*8, self.kernel, stride=2), self.activation())
        
        
        self.fc1 = nn.Identity() if out_shape == self.embed_size else nn.Linear(self.embed_size, out_shape)
        #print("embedsize dell encoder", self.embed_size)
    def forward(self, obs):

        batch_shape = obs.shape[:-3]
        img_shape = obs.shape[-3:]
        #print('obs',obs.shape)
        embed = self.conv1(obs.reshape(-1, *img_shape))
        #print('obs1',embed.shape)
        embed = self.conv2(embed)
        #print('obs2',embed.shape)
        embed = self.conv3(embed)
        #print('obs3',embed.shape)
        if self.game_type == 'atari':
            embed = self.conv4(embed)
        #print('obs4',embed.shape)
        embed = torch.reshape(embed, (*batch_shape, -1))
        #print("embed", embed.shape)
        out = self.fc1(embed)
        return out

    @property
    def embed_size(self):
        if self.game_type == 'minatari':
            conv1_shape = conv_out_shape(self.shape[1:], 0, self.kernel, stride=1)
            conv2_shape = conv_out_shape(conv1_shape, 0, self.kernel, stride=1)
            conv3_shape = conv_out_shape(conv2_shape, 0, self.kernel, stride=1)
            embed_size = int(4*self.depth*np.prod(conv3_shape).item())
        else:
            conv1_shape = conv_out_shape(self.shape[1:], 0, self.kernel, 2)
            conv2_shape = conv_out_shape(conv1_shape, 0, self.kernel, 2)
            conv3_shape = conv_out_shape(conv2_shape, 0, self.kernel, 2)
            conv4_shape = conv_out_shape(conv3_shape, 0, self.kernel, 2)
            embed_size = int(8*self.depth*np.prod(conv4_shape).item())
        
        return embed_size

class ObsDecoder(nn.Module):
    def __init__(self, embed_size, out_shape=(1,64,64), game_type = 'atari', device='cuda'):
        super(ObsDecoder, self).__init__()
        
        self.embed_size = embed_size
        self.activation = nn.ELU
        self.game_type = game_type
        self.out_shape = out_shape
        if game_type == 'minatari':
            self.out_shape = (6,10,10)
            self.depth = 16
            self.kernel = 3
            self.deconv1_pad, self.deconv2_pad, self.deconv3_pad, _, self.conv_shape = self.padding_and_shape
        else:

            self.depth = 128
            self.kernel1, self.kernel2, self.kernel3, self.kernel4 = 5, 5, 6, 6
            self.deconv1_pad, self.deconv2_pad, self.deconv3_pad, self.deconv4_pad, self.conv_shape = self.padding_and_shape

        if self.embed_size == np.prod(self.conv_shape).item():
            self.fc = nn.Identity()
        else:
            self.fc = nn.Linear(self.embed_size, np.prod(self.conv_shape).item())
            self.embed_size = np.prod(self.conv_shape).item()

        if game_type == 'minatari':
            #print('self.embed_size quando creo deconv 1', self.embed_size)
            #self.deconv1 = nn.Sequential(nn.ConvTranspose2d(self.embed_size, self.depth*2, self.kernel), self.activation())
            self.deconv1 = nn.Sequential(nn.ConvTranspose2d(self.depth*4, self.depth*2, self.kernel), self.activation())
            self.deconv2 = nn.Sequential(nn.ConvTranspose2d(self.depth*2, self.depth, self.kernel), self.activation())
            self.deconv3 = nn.Sequential(nn.ConvTranspose2d(self.depth, self.out_shape[0], self.kernel), self.activation())    
        else:    
            self.deconv1 = nn.Sequential(nn.ConvTranspose2d(self.embed_size, self.depth, self.kernel1, stride=2), self.activation())
            self.deconv2 = nn.Sequential(nn.ConvTranspose2d(self.depth, self.depth//2, self.kernel2, stride=2), self.activation())
            self.deconv3 = nn.Sequential(nn.ConvTranspose2d(self.depth//2, self.depth//4, self.kernel3, stride=2), self.activation())
            self.deconv4 = nn.Sequential(nn.ConvTranspose2d(self.depth//4, self.out_shape[0], self.kernel4, stride=2), self.activation())

    def forward(self, embed):
        #print('decoder embed_shape:', embed.shape)
        batch_shape = embed.shape[:-1]
        embed_size = embed.shape[-1]
        prod_size = np.prod(batch_shape).item()
        #print('decoder prod_size: ', prod_size)
        #print('decoder shape before reshape: ', embed.shape)
        embed = embed.reshape(prod_size, embed_size)
        #print('decoder shape after reshape: ', embed.shape)
        embed = self.fc(embed)
        #print('decoder shape after fc: ', embed.shape)
        #print('decoder conv_shape:', self.conv_shape)
        embed = torch.reshape(embed, (prod_size, *self.conv_shape))
        #print('decoder shape after another reshape: ', embed.shape)
        embed = self.deconv1(embed)
        #print('embed1: ', embed.shape)
        embed = self.deconv2(embed)
        #print('embed2: ', embed.shape)
        embed = self.deconv3(embed)
        #print('embed3: ', embed.shape)
        if self.game_type == 'atari':
            embed = self.deconv4(embed)
        #print('embed4: ', embed.shape)
        mean = torch.reshape(embed, (*batch_shape, *self.out_shape))
        obs_dist = td.Independent(td.Normal(mean, 1), len(self.out_shape))
        return obs_dist

    @property
    def padding_and_shape(self):
        if self.game_type == 'minatari':
            deconv3_shape = conv_out_shape(self.out_shape[1:], 0, self.kernel, stride=1)
            deconv3_pad = output_padding_shape(self.out_shape[1:], deconv3_shape, 0, self.kernel, stride=1)
            deconv2_shape = conv_out_shape(deconv3_shape, 0, self.kernel, stride=1)
            deconv2_pad = output_padding_shape(deconv3_shape, deconv2_shape, 0, self.kernel, stride=1)
            deconv1_shape = conv_out_shape(deconv2_shape, 0, self.kernel, stride=1)
            deconv1_pad = output_padding_shape(deconv2_shape, deconv1_shape, 0, self.kernel, stride=1)

            conv_shape = (4 * self.depth, *deconv1_shape) # 4 * 16 = 64
            return deconv1_pad, deconv2_pad, deconv3_pad, None, conv_shape
        else:
            deconv4_shape = conv_out_shape(self.out_shape[1:], 0, self.kernel4, stride=2)
            deconv4_pad = output_padding_shape(self.out_shape[1:], deconv4_shape, 0, self.kernel4, stride=2)
            deconv3_shape = conv_out_shape(deconv4_shape, 0, self.kernel3, stride=2)
            deconv3_pad = output_padding_shape(deconv4_shape, deconv3_shape, 0, self.kernel3, stride=2)
            deconv2_shape = conv_out_shape(deconv3_shape, 0, self.kernel2, stride=2)
            deconv2_pad = output_padding_shape(deconv3_shape, deconv2_shape, 0, self.kernel2, stride=2)
            deconv1_shape = conv_out_shape(deconv2_shape, 0, self.kernel1, stride=2)
            deconv1_pad = output_padding_shape(deconv2_shape, deconv1_shape, 0, self.kernel1, stride=2)

            conv_shape = (8 * self.depth, *deconv1_shape) # 8 * 128 = 1024
            return deconv1_pad, deconv2_pad, deconv3_pad, deconv4_pad, conv_shape
        
        
        
        
        