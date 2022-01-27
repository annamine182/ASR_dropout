from torch import nn

import torch
import torch.nn.functional as F

import espresso.tools.utils as speech_utils

from fairseq.modules import FairseqDropout

class ConvBNReLU(nn.Module):
    """Sequence of convolution-BatchNorm-ReLU layers."""
    def __init__(self, out_channels, kernel_sizes, strides, in_channels=1):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.in_channels = in_channels

        num_layers = len(out_channels)
        assert num_layers == len(kernel_sizes) and num_layers == len(strides)

        self.main_convolutions = nn.ModuleList()
        self.convolutions1 = nn.ModuleList()
        self.convolutions2 = nn.ModuleList()
        self.convolutions3 = nn.ModuleList()
        self.main_batchnorms = nn.ModuleList()
        self.batchnorms1 = nn.ModuleList()
        self.batchnorms2 = nn.ModuleList()
        self.batchnorms3 = nn.ModuleList()
        
        dropout_in = 0.3
        self.dropout_in_module = FairseqDropout(
            dropout_in, module_name=self.__class__.__name__
        )
        dilation_factors = [1, 2, 2, 2]

        for i in range(num_layers):
            self.main_convolutions.append(
                Convolution2d(
                    self.in_channels if i == 0 else self.out_channels[i-1],
                    self.out_channels[i],
                    self.kernel_sizes[i], self.strides[i]))
            self.main_batchnorms.append(nn.BatchNorm2d(out_channels[i]))

        for i in range(num_layers):
            self.convolutions1.append(
                Convolution2d(
                    self.in_channels if i == 0 else self.out_channels[i-1],
                    self.out_channels[i],
                    self.kernel_sizes[i], self.strides[i]))
            self.batchnorms1.append(nn.BatchNorm2d(out_channels[i]))

        dilation_factors = [1,2,2,2]

        for i in range(num_layers):
            self.convolutions2.append(
                Convolution2d(
                    self.in_channels if i == 0 else self.out_channels[i-1],
                    self.out_channels[i],
                    self.kernel_sizes[i], self.strides[i]))
            self.batchnorms2.append(nn.BatchNorm2d(out_channels[i]))

        for i in range(num_layers):
            self.convolutions3.append(
                Convolution2d(
                    self.in_channels if i == 0 else self.out_channels[i-1],
                    self.out_channels[i],
                    self.kernel_sizes[i], self.strides[i]))
            self.batchnorms3.append(nn.BatchNorm2d(out_channels[i]))
        self.frac = nn.Parameter(torch.rand(1))

    def output_lengths(self, in_lengths):
        out_lengths = in_lengths
        for stride in self.strides:
            if isinstance(stride, (list, tuple)):
                assert len(stride) > 0
                s = stride[0]
            else:
                assert isinstance(stride, int)
                s = stride
            out_lengths = (out_lengths + s - 1) // s
        return out_lengths

    def forward(self, src, src_lengths):
        # B X T X C -> B X (input channel num) x T X (C / input channel num)
        x = src.view(
            src.size(0), src.size(1), self.in_channels, src.size(2) // self.in_channels,
        ).transpose(1, 2)
        counter = 0
        #aux_stream = self.dropout_in_module(x)
        x1 = x[:,:,:,:30]
        x2 = x[:,:,:,30:60]
        x3 = x[:,:,:,60:83]
        #print("x1{},x2{},x3{}".format(x1.shape,x2.shape,x3.shape))

        for main_conv, conv1, conv2,conv3, main_bn, bn1, bn2, bn3 in zip(self.main_convolutions, self.convolutions1, self.convolutions2, self.convolutions3, self.main_batchnorms, self.batchnorms1,self.batchnorms2, self.batchnorms3):
            counter+=1
            x = F.relu(main_bn(main_conv(x)))
            x1 = F.relu(bn1(conv1(x1)))
            x2 = F.relu(bn2(conv2(x2)))
            x3 = F.relu(bn3(conv3(x3)))
            #print("x1{},x2{},x3{}".format(x1.shape,x2.shape,x3.shape))
        #print("x1{},x2{},x3{}".format(x1.shape,x2.shape,x3.shape))

            #aux_stream = F.relu(bn2(conv2(aux_stream)))
        x_ = torch.cat((x1,x2,x3),dim=3)
        x_ = x_[:,:,:,:21]
        #print("x1{},x2{},x3{},x{}".format(x1.shape,x2.shape,x3.shape,x.shape))
        x = x+x_
        # B X (output channel num) x T X C' -> B X T X (output channel num) X C'
        x = x.transpose(1, 2)
        # B X T X (output channel num) X C' -> B X T X C
        x = x.contiguous().view(x.size(0), x.size(1), x.size(2) * x.size(3))

        x_lengths = self.output_lengths(src_lengths)
        padding_mask = ~speech_utils.sequence_mask(x_lengths, x.size(1))
        if padding_mask.any():
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        #aux_stream = aux_stream.transpose(1, 2)
        # B X T X (output channel num) X C' -> B X T X C
        #aux_stream = aux_stream.contiguous().view(aux_stream.size(0), aux_stream.size(1), aux_stream.size(2) * aux_stream.size(3))

        #x_lengths = self.output_lengths(src_lengths)
        #padding_mask = ~speech_utils.sequence_mask(x_lengths, aux_stream.size(1))
        #if padding_mask.any():
        #    aux_stream = aux_stream.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        #x = x + aux_stream

        #similarity = F.cosine_similarity(x,dilated_stream,dim=2)
        #torch.set_printoptions(profile="full")
        #torch.set_printoptions(precision=10)
        #print("similarity matrix {} similarity {}".format(similarity.shape, similarity))
        # for i in range(similarity.shape[0]):
        #     for j in range(similarity.shape[1]):
        #         print(similarity[i][j])
        #print("cnn output x{}".format(x.shape))
        return x, x_lengths, padding_mask



def Convolution2d(in_channels, out_channels, kernel_size, stride):
    if isinstance(kernel_size, (list, tuple)):
        if len(kernel_size) != 2:
            assert len(kernel_size) == 1
            kernel_size = (kernel_size[0], kernel_size[0])
    else:
        assert isinstance(kernel_size, int)
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, (list, tuple)):
        if len(stride) != 2:
            assert len(stride) == 1
            stride = (stride[0], stride[0])
    else:
        assert isinstance(stride, int)
        stride = (stride, stride)
    assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1
    padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
    m = nn.Conv2d(
        in_channels, out_channels, kernel_size, stride=stride, padding=padding,
    )
    return m


def DilatedConvolution2d(in_channels, out_channels, kernel_size, stride, dilation):
    if isinstance(kernel_size, (list, tuple)):
        if len(kernel_size) != 2:
            assert len(kernel_size) == 1
            kernel_size = (kernel_size[0], kernel_size[0])
    else:
        assert isinstance(kernel_size, int)
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, (list, tuple)):
        if len(stride) != 2:
            assert len(stride) == 1
            stride = (stride[0], stride[0])
    else:
        assert isinstance(stride, int)
        stride = (stride, stride)
    assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1
    padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
    m = nn.Conv2d(
        in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
    )
    return m
