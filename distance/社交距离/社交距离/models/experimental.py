from models.common import *


class Sum(nn.Module):
    # 多个层加权求和
    def __init__(self, n, weight=False):
        super(Sum, self).__init__()
        self.weight = weight
        self.iter = range(n - 1)
        if weight:
            self.w = nn.Parameter(-torch.arange(1., n) / 2, requires_grad=True)

    def forward(self, x):
        y = x[0]
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


class GhostConv(nn.Module):
    # Ghost Conv
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super(GhostConv, self).__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)


class GhostBottleneck(nn.Module):
    def __init__(self, c1, c2, k, s):  # 如果s==1，则c1=c2，才能shortcut
        super(GhostBottleneck, self).__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1),
                                  DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),
                                  GhostConv(c_, c2, 1, 1, act=False))
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                      Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        y = self.conv(x) + self.shortcut(x)
        return y

class GhostBottleneckAT(nn.Module):
    def __init__(self, c1, c2, k, s, attention='se'):  # 如果s==1，则c1=c2，才能shortcut
        super(GhostBottleneckAT, self).__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1),
                                  DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),
                                  GhostConv(c_, c2, 1, 1, act=False))
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                      Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        print(attention)
        if attention == 'se':
            self.att = SELayer(c2, 16)
        if attention == 'eca':
            self.att = ECALayer(3)


    def forward(self, x):
        y = self.conv(x) + self.shortcut(x)
        return self.att(y)



class ConvPlus(nn.Module):
    # 用扁平卷积核相加
    def __init__(self, c1, c2, k=3, s=1, g=1, bias=True):
        super(ConvPlus, self).__init__()
        self.cv1 = nn.Conv2d(c1, c2, (k, 1), s, (k // 2, 0), groups=g, bias=bias)
        self.cv2 = nn.Conv2d(c1, c2, (1, k), s, (0, k // 2), groups=g, bias=bias)

    def forward(self, x):
        return self.cv1(x) + self.cv2(x)


class MixConv2d(nn.Module):
    #
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):
        super(MixConv2d, self).__init__()
        groups = len(k)
        if equal_ch:
            i = torch.linspace(0, groups - 1E-6, c2).floor()
            c_ = [(i == g).sum() for g in range(groups)]
        else:
            b = [c2] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()

        self.m = nn.ModuleList([nn.Conv2d(c1, int(c_[g]), k[g], s, k[g] // 2, bias=False) for g in range(groups)])
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return x + self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))