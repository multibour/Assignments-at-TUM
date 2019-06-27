"""SegmentationNN"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.decode(x)

##################################################################################################################

class conv_block(nn.Module):
    def __init__(self, in_map, out_map, kernel=3, stride=1, activation=True):
        super(conv_block, self).__init__()

        self._mconv = nn.Sequential(
            nn.Conv2d(in_map, out_map, kernel, stride, (kernel) // 2),
            nn.BatchNorm2d(out_map)
        )

        if (activation):
            self._mconv.add_module("conv_block_relu", nn.ReLU(inplace=True))

    def forward(self, x):
        out = self._mconv(x)

        return out


class deconv_block(nn.Module):
    def __init__(self, in_map, out_map, kernel=3, stride=2, padding=1):
        super(deconv_block, self).__init__()

        self._conv_trans_2d = nn.ConvTranspose2d(in_map, out_map, kernel, stride, padding)
        self._batch_norm_2d = nn.BatchNorm2d(out_map)
        self._relu = nn.ReLU(inplace=True)

    def forward(self, x, output_size):
        out = self._conv_trans_2d(x, output_size=output_size)

        return out


class res_block(nn.Module):
    def __init__(self, in_map, out_map, downsample=False):
        super(res_block, self).__init__()

        self._mconv_2 = conv_block(out_map, out_map, 3, 1, False)

        if downsample == True:
            stride = 2
        else:
            stride = 1

        self._mconv_1 = conv_block(in_map, out_map, 3, stride)
        self._mdownsample = nn.Sequential(
            nn.Conv2d(in_map, out_map, 1, stride),
            nn.BatchNorm2d(out_map)
        )
        self._relu = nn.ReLU(inplace=True)

    def forward(self, x):

        residual = x
        out = self._mconv_1(x)
        out = self._mconv_2(out)
        residual = self._mdownsample(x)
        # print("residual size,", residual.size())
        out = residual + out
        out = self._relu(out)

        return out


class encoder(nn.Module):
    def __init__(self, in_map, out_map):
        super(encoder, self).__init__()

        self._mres_1 = res_block(in_map, out_map, True)
        self._mres_2 = res_block(out_map, out_map)

    def forward(self, x):
        out = self._mres_1(x)
        out = self._mres_2(out)

        return out


class decoder(nn.Module):
    def __init__(self, in_map, out_map, padding=1):
        super(decoder, self).__init__()

        self._mconv_1 = conv_block(in_map, in_map // 4, 1)
        self._mdeconv_1 = deconv_block(in_map // 4, in_map // 4, 3, 2, padding)
        self._mconv_2 = conv_block(in_map // 4, out_map, 1)

    def forward(self, x, output_size):
        out = self._mconv_1(x)
        out = self._mdeconv_1(out, output_size=output_size)
        out = self._mconv_2(out)

        return out

##################################################################################################################


class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23):
        super(SegmentationNN, self).__init__()

        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################
        self._mconv_1 = conv_block(3, 64, 7, 2)
        self._mmax_pool = nn.MaxPool2d(3, 2, padding=1)

        self._mencoder_1 = encoder(64, 64)
        self._mencoder_2 = encoder(64, 128)
        self._mencoder_3 = encoder(128, 256)
        self._mencoder_4 = encoder(256, 512)

        self._mdecoder_1 = decoder(64, 64)
        self._mdecoder_2 = decoder(128, 64)
        self._mdecoder_3 = decoder(256, 128)
        self._mdecoder_4 = decoder(512, 256)

        self._deconv_1 = deconv_block(64, 32)
        self._mconv_2 = conv_block(32, 32, 3)
        self._deconv_2 = deconv_block(32, num_classes, 2, 2, 0)
        '''
        self.enc1 = _EncoderBlock(3, 64)
        self.enc2 = _EncoderBlock(64, 128)
        self.enc3 = _EncoderBlock(128, 256)
        self.enc4 = _EncoderBlock(256, 512, dropout=True)
        self.center = _DecoderBlock(512, 1024, 512)
        self.dec4 = _DecoderBlock(1024, 512, 256)
        self.dec3 = _DecoderBlock(512, 256, 128)
        self.dec2 = _DecoderBlock(256, 128, 64)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        '''
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################
        conv_down_out = self._mconv_1(x)
        max_pool_out = self._mmax_pool(conv_down_out)

        encoder_1_out = self._mencoder_1(max_pool_out)
        encoder_2_out = self._mencoder_2(encoder_1_out)
        encoder_3_out = self._mencoder_3(encoder_2_out)
        encoder_4_out = self._mencoder_4(encoder_3_out)

        decoder_4_out = self._mdecoder_4(encoder_4_out, encoder_3_out.size()) + encoder_3_out
        decoder_3_out = self._mdecoder_3(decoder_4_out, encoder_2_out.size()) + encoder_2_out
        decoder_2_out = self._mdecoder_2(decoder_3_out, encoder_1_out.size()) + encoder_1_out
        decoder_1_out = self._mdecoder_1(decoder_2_out, max_pool_out.size())

        deconv_out = self._deconv_1(decoder_1_out, conv_down_out.size())
        conv_2_out = self._mconv_2(deconv_out)
        x = self._deconv_2(conv_2_out, x.size())
        '''
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(enc4)
        dec4 = self.dec4(torch.cat([center, F.upsample(enc4, center.size()[2:], mode='bilinear')], 1))
        dec3 = self.dec3(torch.cat([dec4, F.upsample(enc3, dec4.size()[2:], mode='bilinear')], 1))
        dec2 = self.dec2(torch.cat([dec3, F.upsample(enc2, dec3.size()[2:], mode='bilinear')], 1))
        dec1 = self.dec1(torch.cat([dec2, F.upsample(enc1, dec2.size()[2:], mode='bilinear')], 1))
        final = self.final(dec1)

        return F.upsample(final, x.size()[2:], mode='bilinear')
        '''
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
