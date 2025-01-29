from .ResNet import *
from .ResNets import *
from .VGG import *
from .VGG_LTH import *
from .MedMNIST_ResNet import *

model_dict = {
    'medmnist_resnet18': ResNet18,
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet20s': resnet20s,
    'resnet44s': resnet44s,
    'resnet56s': resnet56s,
    'vgg16_bn': vgg16_bn,
    'vgg16_bn_lth': vgg16_bn_lth,
}
