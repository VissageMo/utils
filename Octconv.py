import torch
import torch.nn as nn
import torch.nn.functional as F


up_kwargs = {'mode': 'nearest'}


class OctConv(nn.Module):
    """ 
    Octivate conv layer for model compression.

    Args:
        alpha_in: compression ratio of input channel, range(0, 1)
        alpha_out: comression ratio of current channel, range(0, 1)
        up_kwargs: mode of upsample, include ['nearest', 'bilinear']
    Forward:
        x: [x_h, x_l] input image of both high frequency and low frequency
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, alpha_in=0.5, alpha_out=0.5, 
                    stride=1, padding=1, dilation=1, groups=1, bias=False, up_kwargs=up_kwargs):
        super(OctConv, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = torch.zeros(out_channels).cuda()
        self.up_kwargs = up_kwargs
        self.pool = nn.AvgPool2d(kennel_size=(2, 2), stride=2)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out

    def forward(self, x):
        xh, xl = x
        if self.stride == 2:
            xh, xl = self.pool(xh), self.pool(xl)
        x_h2l = self.pool(xh)

        # high-[0:split_in] low-[split_out:]
        split_in = int(self.in_channels * (1 - self.alpha_in)) 
        split_out = int(self.out_channels * (1 - self.alpha_out))

        x_h2h = F.conv2d(xh, self.weights[:split_out, :split_in, :, :], self.bias[:split_out], 1,
                            self.padding, self.dilation, self.groups)
        x_l2l = F.conv2d(xl, self.weights[split_out:, split_in:, :, :], self.bias[split_out:], 1,
                            self.padding, self.dilation, self.groups)

        x_h2l = F.conv2d(x_h2l, self.weights[split_out:, :split_in, :, :], self.bias[split_out:], 1,
                            self.padding, self.dilation, self.groups)
        x_l2h = F.conv2d(xl, self.weights[:split_out, split_in:, :, :], self.bias[:split_out], 1,
                            self.padding, self.dilation, self.groups)
        x_l2h = F.upsample(x_l2h, scale_factor=2, **self.up_kwargs)

        X_h = x_h2h + x_l2h
        X_l = x_l2l + x_h2l

        return X_h, X_l


class firstOct(nn.Module):
    """
    Octivate conv for input layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, alpha_in=0, alpha_out=0.5,
                    stride=1, padding=1, dilation=1, groups=1, bias=False, up_kwargs=up_kwargs):
        super(firstOct, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = torch.zeros(out_channels).cuda()
        self.up_kwargs = up_kwargs
        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out

    def forward(self, x):
        xh = x
        xl = self.pool(xh)
        if self.stride == 2:
            xh, xl = self.pool(xh), self.pool(xl)
        
        # high-[0:split_in] low-[split_out:]
        split_in = int(self.in_channels * (1 - self.alpha_in)) 
        split_out = int(self.out_channels * (1 - self.alpha_out))

        x_h2h = F.conv2d(xh, self.weights[:split_out, :split_in, :, :], self.bias[:split_out], 1,
                            self.padding, self.dilation, self.groups)
        x_l2l = F.conv2d(xl, self.weights[split_out:, :split_in, :, :], self.bias[split_out:], 1,
                            self.padding, self.dilation, self.groups)

        X_h, X_l = x_h2h, x_l2l
        
        return X_h, X_l


class lastOct(nn.Module):
    """
    Octivate conv for output layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, alpha_in=0.5, alpha_out=0,
                    stride=1, padding=1, dilation=1, groups=1, bias=False, up_kwargs=up_kwargs):
        super(firstOct, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = torch.zeros(out_channels).cuda()
        self.up_kwargs = up_kwargs
        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out

    def forward(self, x):
        xh, xl = x
        if self.stride == 2:
            xh, xl = self.pool(xh), self.pool(xl)
        
        # high-[0:split_in] low-[split_out:]
        split_in = int(self.in_channels * (1 - self.alpha_in)) 
        split_out = int(self.out_channels * (1 - self.alpha_out))

        x_h2h = F.conv2d(xh, self.weights[:split_out, :split_in, :, :], self.bias[:split_out], 1,
                            self.padding, self.dilation, self.groups)
        x_l2h = F.conv2d(xl, self.weights[:split_out, split_in:, :, :], self.bias[:split_out], 1,
                            self.padding, self.dilation, self.groups)

        X_h = x_h2h + x_l2h
        
        return X_h


class OctLayerCBR(nn.Module):
    """
    """
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), alpha_in=0.5, alpha_out=0.5, 
                    stride=1, padding=1, dilation=1, groups=1, bias=False, up_kwargs=up_kwargs, norm_layer=nn.BatchNorm2d):
        super(OctLayerCBR, self).__init__()
        self.conv = OctConv(in_channels, out_channels, kernel_size, alpha_in, alpha_out,
                            stride, padding, dilation, groups, bias, up_kwargs)
        self.bn_h = norm_layer(out_channels * (1 - alpha_out))
        self.bn_l = norm_layer(out_channels * alpha_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        xh, xl = self.conv(x)
        xh = self.relu(self.bn_h(xh))
        xl = self.relu(self.bn_l(xl))

        return xh, xl


class OctLayerCB(nn.Module):
    """
    """
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), alpha_in=0.5, alpha_out=0.5, 
                    stride=1, padding=1, dilation=1, groups=1, bias=False, up_kwargs=up_kwargs, norm_layer=nn.BatchNorm2d):
        super(OctLayerCB, self).__init__()
        self.conv = OctConv(in_channels, out_channels, kernel_size, alpha_in, alpha_out,
                            stride, padding, dilation, groups, bias, up_kwargs)
        self.bn_h = norm_layer(out_channels * (1 - alpha_out))
        self.bn_l = norm_layer(out_channels * alpha_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        xh, xl = self.conv(x)
        xh = self.bn_h(xh)
        xl = self.bn_l(xl)

        return xh, xl


class firstOctLayerCBR(nn.Module):
    """
    """
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), alpha_in=0.5, alpha_out=0.5, 
                    stride=1, padding=1, dilation=1, groups=1, bias=False, up_kwargs=up_kwargs, norm_layer=nn.BatchNorm2d):
        super(firstOctLayerCBR, self).__init__()
        self.conv = firstOct(in_channels, out_channels, kernel_size, alpha_in, alpha_out,
                            stride, padding, dilation, groups, bias, up_kwargs)
        self.bn_h = norm_layer(out_channels * (1 - alpha_out))
        self.bn_l = norm_layer(out_channels * alpha_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        xh, xl = self.conv(x)
        xh = self.relu(self.bn_h(xh))
        xl = self.relu(self.bn_l(xl))

        return xh, xl


class firstOctLayerCB(nn.Module):
    """
    """
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), alpha_in=0.5, alpha_out=0.5, 
                    stride=1, padding=1, dilation=1, groups=1, bias=False, up_kwargs=up_kwargs, norm_layer=nn.BatchNorm2d):
        super(firstOctLayerCB, self).__init__()
        self.conv = firstOct(in_channels, out_channels, kernel_size, alpha_in, alpha_out,
                            stride, padding, dilation, groups, bias, up_kwargs)
        self.bn_h = norm_layer(out_channels * (1 - alpha_out))
        self.bn_l = norm_layer(out_channels * alpha_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        xh, xl = self.conv(x)
        xh = self.bn_h(xh)
        xl = self.bn_l(xl)

        return xh, xl


class lastOctLayerCBR(nn.Module):
    """
    """
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), alpha_in=0.5, alpha_out=0.5, 
                    stride=1, padding=1, dilation=1, groups=1, bias=False, up_kwargs=up_kwargs, norm_layer=nn.BatchNorm2d):
        super(lastOctLayerCBR, self).__init__()
        self.conv = firstOct(in_channels, out_channels, kernel_size, alpha_in, alpha_out,
                            stride, padding, dilation, groups, bias, up_kwargs)
        self.bn_h = norm_layer(out_channels * (1 - alpha_out))
        self.bn_l = norm_layer(out_channels * alpha_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        xh, xl = self.conv(x)
        xh = self.relu(self.bn_h(xh))
        xl = self.relu(self.bn_l(xl))

        return xh, xl


class lastOctLayerCB(nn.Module):
    """
    """
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), alpha_in=0.5, alpha_out=0.5, 
                    stride=1, padding=1, dilation=1, groups=1, bias=False, up_kwargs=up_kwargs, norm_layer=nn.BatchNorm2d):
        super(lastOctLayerCB, self).__init__()
        self.conv = firstOct(in_channels, out_channels, kernel_size, alpha_in, alpha_out,
                            stride, padding, dilation, groups, bias, up_kwargs)
        self.bn_h = norm_layer(out_channels * (1 - alpha_out))
        self.bn_l = norm_layer(out_channels * alpha_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        xh, xl = self.conv(x)
        xh = self.bn_h(xh)
        xl = self.bn_l(xl)

        return xh, xl


if __name__ == '__main__':
    # nn.Conv2d
    high = torch.Tensor(1, 64, 32, 32).cuda()
    low = torch.Tensor(1, 192, 16, 16).cuda()
    # test Oc conv
    OCconv = OctConv(kernel_size=(3,3),in_channels=256,out_channels=512,bias=False,stride=2,alpha_in=0.75,alpha_out=0.75).cuda()
    i = high,low
    x_out,y_out = OCconv(i)
    print(x_out.size())
    print(y_out.size())
    # test First Octave Cov
    i = torch.Tensor(1, 3, 512, 512).cuda()
    FOCconv = firstOct(kernel_size=(3,3), in_channels=3, out_channels=128).cuda()
    x_out, y_out = FOCconv(i)
    # test last Octave Cov
    LOCconv = lastOct(kernel_size=(3,3), in_channels=256, out_channels=128, alpha_out=0.75, alpha_in=0.75).cuda()
    i = high, low
    out = LOCconv(i)
    print(out.size())
    # test OCB
    ocb = OctLayerCB(in_channels=256, out_channels=128, alpha_out=0.75, alpha_in=0.75).cuda()
    i = high, low
    x_out_h, y_out_l = ocb(i)
    print(x_out_h.size())
    print(y_out_l.size())

    ocb_last = lastOctLayerCBR(256,128, alpha_out=0.0, alpha_in=0.75).cuda()
    i = high, low
    x_out_h = ocb_last(i)
    print(x_out_h.size())