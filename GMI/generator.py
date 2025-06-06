import torch
import torch.nn as nn

class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, bn=False, stride=1):
        super(ResBlock, self).__init__()
        self.bn = bn
        if bn:
            self.bn0 = nn.BatchNorm2d(in_planes)

        
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1)
        if bn:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        if self.bn:
            out = F.relu(self.bn0(x))
        else:
            out = F.relu(x)

        if self.bn:
            out = F.relu(self.bn1(self.conv1(out)))
        else:
            out = F.relu(self.conv1(out))

        out = self.conv2(out)
        out += self.shortcut(x)
        # out = F.relu(out)
        return out
        
class res_normN_AE(nn.Module):
    def __init__(self, N = 0, internal_nc = 64, input_nc=256, output_nc=3, input_dim=8, output_dim=32, activation = "sigmoid"):
        super(res_normN_AE, self).__init__()
        if input_dim != 0:
            upsampling_num = int(np.log2(output_dim // input_dim)) # input_dim =0 denotes confidensce score
            self.confidence_score = False
        else:
            upsampling_num = int(np.log2(output_dim))
            self.confidence_score = True
        model = []
           
        model += [ResBlock(input_nc, internal_nc, bn = True, stride=1)] #first
        model += [nn.ReLU()]
        # model += [nn.Dropout(0.2)] 

        for _ in range(N):
            model += [ResBlock(internal_nc, internal_nc, bn = True, stride=1)]
            # model += [nn.Sigmoid()]
            model += [nn.ReLU()]
        # model += [nn.Dropout(0.25)]

        if upsampling_num >= 1:
            model += [nn.ConvTranspose2d(internal_nc, internal_nc, kernel_size=3, stride=2, padding=1, output_padding=1)]
            model += [nn.BatchNorm2d(internal_nc)]
        else:
            model += [ResBlock(internal_nc, internal_nc, bn = True, stride=1)] #second
        model += [nn.ReLU()]

        if upsampling_num >= 2:
            model += [nn.ConvTranspose2d(internal_nc, internal_nc, kernel_size=3, stride=2, padding=1, output_padding=1)]
            model += [nn.BatchNorm2d(internal_nc)]
        else:
            model += [ResBlock(internal_nc, internal_nc, bn = True, stride=1)]
        model += [nn.ReLU()]

        if upsampling_num >= 3:
            for _ in range(upsampling_num - 2):
                model += [nn.ConvTranspose2d(internal_nc, internal_nc, kernel_size=3, stride=2, padding=1, output_padding=1)]
                model += [nn.BatchNorm2d(internal_nc)]
                model += [nn.ReLU()]

        model += [ResBlock(internal_nc, output_nc, bn = True, stride=1)]
        if activation == "sigmoid":
            model += [nn.Sigmoid()]
            # model += [nn.Tanh()]
        elif activation == "tanh":
            model += [nn.Tanh()]
        self.m = nn.Sequential(*model)

    def forward(self, x):
        if self.confidence_score:
            x = x.view(x.size(0), x.size(2), 1, 1)
        output = self.m(x)
        return output
        
class Generator(nn.Module):
    def __init__(self, in_dim=100, dim=64):
        super(Generator, self).__init__()

        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 5, 2,
                                   padding=2, output_padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU())

        self.l1 = nn.Sequential(
            nn.Linear(in_dim, dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.ReLU())
        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 8, dim * 4),
            dconv_bn_relu(dim * 4, dim * 2),
            dconv_bn_relu(dim * 2, dim),
            nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1),
            nn.Sigmoid())

    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2_5(y)
        return y

class GeneratorMNIST(nn.Module):
    def __init__(self, in_dim=100, dim=64):
        super(GeneratorMNIST, self).__init__()

        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 5, 2,
                                   padding=2, output_padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU())

        self.l1 = nn.Sequential(
            nn.Linear(in_dim, dim * 4 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 4 * 4 * 4),
            nn.ReLU())
        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 4, dim * 2),
            dconv_bn_relu(dim * 2, dim),
            nn.ConvTranspose2d(dim, 1, 5, 2, padding=2, output_padding=1),
            nn.Sigmoid())

    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2_5(y)
        return y


class GeneratorCIFAR(nn.Module):    
    ### the output shape will be bt*3*32*32  
    def __init__(self, in_dim, dim=64):
        super(GeneratorCIFAR, self).__init__()
        self.in_dim = in_dim
        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 5, 2,
                                   padding=2, output_padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU())

        self.l1 = nn.Sequential(
            nn.Linear(in_dim, dim * 4 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 4 * 4 * 4),
            nn.ReLU())
        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 4, dim * 2),
            dconv_bn_relu(dim * 2, dim),
            nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1),
            nn.Sigmoid())

    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2_5(y)
        return y
    

class CompletionNetwork(nn.Module):
    def __init__(self):
        super(CompletionNetwork, self).__init__()
        # input_shape: (None, 4, img_h, img_w)
        self.conv1 = nn.Conv2d(4, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.ReLU()
        # input_shape: (None, 64, img_h, img_w)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.act2 = nn.ReLU()
        # input_shape: (None, 128, img_h//2, img_w//2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.act3 = nn.ReLU()
        # input_shape: (None, 128, img_h//2, img_w//2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.act4 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.act5 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.act6 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, stride=1, dilation=2, padding=2)
        self.bn7 = nn.BatchNorm2d(128)
        self.act7 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, stride=1, dilation=4, padding=4)
        self.bn8 = nn.BatchNorm2d(128)
        self.act8 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv9 = nn.Conv2d(128, 128, kernel_size=3, stride=1, dilation=8, padding=8)
        self.bn9 = nn.BatchNorm2d(128)
        self.act9 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv10 = nn.Conv2d(128, 128, kernel_size=3, stride=1, dilation=16, padding=16)
        self.bn10 = nn.BatchNorm2d(128)
        self.act10 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv11 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn11 = nn.BatchNorm2d(128)
        self.act11 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv12 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn12 = nn.BatchNorm2d(128)
        self.act12 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.deconv13 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn13 = nn.BatchNorm2d(64)
        self.act13 = nn.ReLU()
        # input_shape: (None, 128, img_h//2, img_w//2)
        self.conv14 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn14 = nn.BatchNorm2d(64)
        self.act14 = nn.ReLU()
        # input_shape: (None, 128, img_h//2, img_w//2)
        self.deconv15 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn15 = nn.BatchNorm2d(32)
        self.act15 = nn.ReLU()
        # input_shape: (None, 64, img_h, img_w)
        self.conv16 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn16 = nn.BatchNorm2d(32)
        self.act16 = nn.ReLU()
        # input_shape: (None, 32, img_h, img_w)
        self.conv17 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        self.act17 = nn.Sigmoid()
        # output_shape: (None, 3, img_h. img_w)

    def forward(self, x):
        x = self.bn1(self.act1(self.conv1(x)))
        x = self.bn2(self.act2(self.conv2(x)))
        x = self.bn3(self.act3(self.conv3(x)))
        x = self.bn4(self.act4(self.conv4(x)))
        x = self.bn5(self.act5(self.conv5(x)))
        x = self.bn6(self.act6(self.conv6(x)))
        x = self.bn7(self.act7(self.conv7(x)))
        x = self.bn8(self.act8(self.conv8(x)))
        x = self.bn9(self.act9(self.conv9(x)))
        x = self.bn10(self.act10(self.conv10(x)))
        x = self.bn11(self.act11(self.conv11(x)))
        x = self.bn12(self.act12(self.conv12(x)))
        x = self.bn13(self.act13(self.deconv13(x)))
        x = self.bn14(self.act14(self.conv14(x)))
        x = self.bn15(self.act15(self.deconv15(x)))
        x = self.bn16(self.act16(self.conv16(x)))
        x = self.act17(self.conv17(x))
        return x


def dconv_bn_relu(in_dim, out_dim):
    return nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, 5, 2,
                           padding=2, output_padding=1, bias=False),
        nn.BatchNorm2d(out_dim),
        nn.ReLU())


class ContextNetwork(nn.Module):
    def __init__(self):
        super(ContextNetwork, self).__init__()
        # input_shape: (None, 4, img_h, img_w)
        self.conv1 = nn.Conv2d(4, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.ReLU()
        # input_shape: (None, 32, img_h, img_w)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.act2 = nn.ReLU()
        # input_shape: (None, 64, img_h//2, img_w//2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.act3 = nn.ReLU()
        # input_shape: (None, 128, img_h//2, img_w//2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.act4 = nn.ReLU()
        # input_shape: (None, 128, img_h//4, img_w//4)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.act5 = nn.ReLU()
        # input_shape: (None, 128, img_h//4, img_w//4)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.act6 = nn.ReLU()
        # input_shape: (None, 128, img_h//4, img_w//4)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, stride=1, dilation=2, padding=2)
        self.bn7 = nn.BatchNorm2d(128)
        self.act7 = nn.ReLU()
        # input_shape: (None, 128, img_h//4, img_w//4)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, stride=1, dilation=4, padding=4)
        self.bn8 = nn.BatchNorm2d(128)
        self.act8 = nn.ReLU()
        # input_shape: (None, 128, img_h//4, img_w//4)
        self.conv9 = nn.Conv2d(128, 128, kernel_size=3, stride=1, dilation=8, padding=8)
        self.bn9 = nn.BatchNorm2d(128)
        self.act9 = nn.ReLU()
        # input_shape: (None, 128, img_h//4, img_w//4)
        self.conv10 = nn.Conv2d(128, 128, kernel_size=3, stride=1, dilation=16, padding=16)
        self.bn10 = nn.BatchNorm2d(128)
        self.act10 = nn.ReLU()

    def forward(self, x):
        x = self.bn1(self.act1(self.conv1(x)))
        x = self.bn2(self.act2(self.conv2(x)))
        x = self.bn3(self.act3(self.conv3(x)))
        x = self.bn4(self.act4(self.conv4(x)))
        x = self.bn5(self.act5(self.conv5(x)))
        x = self.bn6(self.act6(self.conv6(x)))
        x = self.bn7(self.act7(self.conv7(x)))
        x = self.bn8(self.act8(self.conv8(x)))
        x = self.bn9(self.act9(self.conv9(x)))
        x = self.bn10(self.act10(self.conv10(x)))
        return x


class IdentityGenerator(nn.Module):

    def __init__(self, in_dim=100, dim=64):
        super(IdentityGenerator, self).__init__()

        self.l1 = nn.Sequential(
            nn.Linear(in_dim, dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.ReLU())
        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 8, dim * 4),
            dconv_bn_relu(dim * 4, dim * 2))

    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2_5(y)
        return y


class InversionNet(nn.Module):
    def __init__(self, out_dim=128):
        super(InversionNet, self).__init__()

        # input [4, h, w]  output [256, h // 4, w // 4]
        self.ContextNetwork = ContextNetwork()
        # input [z_dim] output[128, 16, 16]
        self.IdentityGenerator = IdentityGenerator()

        self.dim = 128 + 128
        self.out_dim = out_dim

        self.Dconv = nn.Sequential(
            dconv_bn_relu(self.dim, self.out_dim),
            dconv_bn_relu(self.out_dim, self.out_dim // 2))

        self.Conv = nn.Sequential(
            nn.Conv2d(self.out_dim // 2, self.out_dim // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_dim // 4),
            nn.ReLU(),
            nn.Conv2d(self.out_dim // 4, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid())

    def forward(self, inp):
        # x.shape [4, h, w]  z.shape [100]
        x, z = inp
        context_info = self.ContextNetwork(x)
        identity_info = self.IdentityGenerator(z)
        # []
        y = torch.cat((context_info, identity_info), dim=1)
        y = self.Dconv(y)
        y = self.Conv(y)

        return y
