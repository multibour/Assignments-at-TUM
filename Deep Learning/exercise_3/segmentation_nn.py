"""SegmentationNN"""
import torch
import torch.nn as nn
import torchvision.models as models


class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23):
        super(SegmentationNN, self).__init__()

        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################
        print("started")
        padding = 0
        vgg16 = models.vgg16(pretrained=True)
        #for param in vgg16.features.parameters():
            #param.requires_grad = False
        self.vgg16layer = vgg16.features
        self.upsamplelayer = nn.Upsample(size=(240,240))#scale_factor=2)
        self.convlayer = torch.nn.Conv2d(512,num_classes,1,1,padding = padding)
        print("imported")
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
        #print(x.size())
        vgg16 = self.vgg16layer(x)
        #print("vgg16 size:  ")
        #print(vgg16.size())
        upsample = self.upsamplelayer(vgg16)
        #print("upsample size: ")
        #print(upsample.size())
        conv = self.convlayer(upsample)
        #print("conv size")
        #print(conv.size())
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return conv

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
