import numpy as np
import torch
import torch.nn as nn

class VariationalAutoencoder(nn.Module):
    def __init__(self, imageShape, firstFilterCount, act, layerwise=True):
        super(VariationalAutoencoder, self).__init__()
        self.act = act
        self.imageShape = imageShape
        self.firstFilterCount = firstFilterCount
        self.layerwise = layerwise

        self.encoder = VariationalEncoder(imageShape=imageShape, firstFilterCount=firstFilterCount, act=act, layerwise=layerwise)
        self.decoder = VariationalDecoder(imageShape=imageShape, firstFilterCount=firstFilterCount, act=act, layerwise=layerwise)
        self.decoder_segmentation = VariationalDecoder(imageShape=imageShape, firstFilterCount=firstFilterCount, act=act, layerwise=layerwise)

    def forward(self, x, lod, printCode=False):
        lod = lod - 2
        z, mu, logVar, shape = self.encoder.forward(x, lod)
        x_reconstructed = torch.sigmoid(self.decoder.forward(z, lod, shape))
        x_segmentation = torch.sigmoid(self.decoder_segmentation.forward(z, lod, shape, detach=True))
        if printCode:
            print(z)
        return ((x_reconstructed, mu, logVar), x_segmentation)

class VariationalEncoder(nn.Module):
    def __init__(self, imageShape, firstFilterCount, act, layerwise=True):
        super(VariationalEncoder, self).__init__()
        self.act = act
        self.imageShape = imageShape
        self.firstFilterCount = firstFilterCount
        self.layerwise = layerwise

        self.convDownsamplingLayers = torch.nn.ModuleList()
        self.muEncodingLayers = torch.nn.ModuleList()
        self.logVarEncodingLayers = torch.nn.ModuleList()

        for level in range(int(np.log2(self.imageShape[1])-1)):
            if level == 0:
                self.convDownsamplingLayers.append(torch.nn.Conv2d(in_channels=self.imageShape[0], out_channels=firstFilterCount, kernel_size=4, stride=2, padding=1))
            else:
                self.convDownsamplingLayers.append(torch.nn.Conv2d(in_channels=firstFilterCount * 2**(level - 1), out_channels=firstFilterCount * 2**(level), kernel_size=4, stride=2, padding=1))

            features, code_length = self.firstFilterCount * 2 ** (level + 2), int(2 ** (level + 2))
            self.muEncodingLayers.append(torch.nn.Linear(in_features=features, out_features=code_length))
            self.logVarEncodingLayers.append(torch.nn.Linear(in_features=features, out_features=code_length))
    
    def sample(self, mu, logVar):
        # Reparameterize:
        std = torch.exp(0.5 * logVar)
        epsilon = torch.randn_like(std)
        z = mu + std * epsilon
        return z
    
    def encode(self, x, scale):
        for layer in range(scale):
            x = self.convDownsamplingLayers[layer](x)

        # calculate relevant layer of requested scale
        x = self.convDownsamplingLayers[scale](x)

        # VAE: enforcing gaussian prior
        shape = x.shape
        x = torch.flatten(x, start_dim=1)
        mu = (self.muEncodingLayers[scale](x))
        logVar = (self.logVarEncodingLayers[scale](x))
        
        return mu, logVar, shape
    
    def forward(self, x, scale):
        mu, logVar, shape = self.encode(x, scale)
        z = self.sample(mu, logVar)
        return z, mu, logVar, shape

class VariationalDecoder(nn.Module):
    def __init__(self, imageShape, firstFilterCount, act, layerwise=True):
        super(VariationalDecoder, self).__init__()
        self.act = act
        self.imageShape = imageShape
        self.firstFilterCount = firstFilterCount
        self.layerwise = layerwise

        self.convUpsamplingLayers = torch.nn.ModuleList()
        self.zDecodingLayers = torch.nn.ModuleList()

        for level in range(int(np.log2(self.imageShape[1])-1)):
            if level == 0:
                self.convUpsamplingLayers.append(torch.nn.ConvTranspose2d(in_channels=firstFilterCount, out_channels=self.imageShape[0], kernel_size=4, stride=2, padding=1))
            else:
                self.convUpsamplingLayers.append(torch.nn.ConvTranspose2d(in_channels=int(firstFilterCount * 2**(level)), out_channels=int(firstFilterCount * 2**(level - 1)), kernel_size=4, stride=2, padding=1))
            features, code_length = self.firstFilterCount * 2 ** (level + 2), int(2 ** (level + 2))
            self.zDecodingLayers.append(torch.nn.Linear(in_features=code_length, out_features=features))
    
    def decode(self, z, scale, shape):
        x = self.act(self.zDecodingLayers[scale](z)).reshape(shape)
        
        # Transpose Convolutions
        for layer in range(scale):
            x = self.act(self.convUpsamplingLayers[scale-layer](x))
        x = self.convUpsamplingLayers[0](x)

        return x

    def forward(self, z, scale, shape, detach=False):
        if detach:
            z = z.detach()
        x = self.decode(z, scale, shape)
        return x