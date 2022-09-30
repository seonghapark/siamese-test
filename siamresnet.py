from torch import nn
import resnet

class SiamResNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone, self.embedding = resnet.__dict__[args.arch](
            zero_init_residual=True
        )

    def forward(self, x, y):
        x = self.backbone(x)
        y = self.backbone(y)

        return x, y
