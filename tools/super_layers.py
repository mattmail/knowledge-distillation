import torch.nn as nn


class SuperConv(nn.Module):
    """
    This convolution can be Conv2d or Conv3d based on the img_dim 
    parameter. 
    """
    def __init__(self,in_channels, out_channels, img_dim ="3D",
                kernel_size=3, stride=1, padding=1, bias=0):
        super(SuperConv, self).__init__()
        assert(img_dim  in ["3D", "2D"])
        self.conv= None
        if img_dim =="3D":
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size
            , stride=stride, padding=padding, bias=bias)
        else: 
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size
            , stride=stride, padding=padding, bias=bias) 
    def forward(self, x):
        x = self.conv(x)
        return(x)       
              
class SuperInstanceNorm(nn.Module):
    """
    This InstanceNorm can be InstanceNorm2d or InstanceNorm3d based on the img_dim 
    parameter. 
    """
    def __init__(self,number_channels,img_dim = "3D" ):
        super(SuperInstanceNorm, self).__init__()
        assert(img_dim  in ["3D", "2D"])
        self.InstanceNorm = None 
        if img_dim =="3D":
            self.InstanceNorm =  nn.InstanceNorm3d(number_channels)
        else: 
            self.InstanceNorm =  nn.InstanceNorm2d(number_channels)   
    def  forward (self,x):
        return(self.InstanceNorm(x))         
class SuperMaxPool (nn.Module):
    """
    This Maxpool function can be Mxpool2d or Maxpool3d based on the img_dim 
    parameter. 
    """
    def __init__(self,kernel_size=2, padding=0, img_dim = "3D" ):
        super(SuperMaxPool, self).__init__()
        assert(img_dim  in ["3D", "2D"])
        self.maxpool = None 
        if img_dim =="3D":
            self.maxpool =  nn.MaxPool3d(kernel_size=kernel_size, padding=padding)
        else: 
            self.maxpool =  nn.MaxPool2d(kernel_size=kernel_size, padding=padding)
    def forward(self,x):
        return(self.maxpool(x))      




















