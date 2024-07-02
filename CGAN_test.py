import jittor as jt
from jittor import init
import argparse
import os
import numpy as np
import math
from jittor import nn
from PIL import Image

if jt.has_cuda:
    jt.flags.use_cuda = 1

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--n_classes', type=int, default=10, help='number of classes for dataset')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=1000, help='interval between image sampling')
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)
        # nn.Linear(in_dim, out_dim)表示全连接层
        # in_dim：输入向量维度
        # out_dim：输出向量维度
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            return layers
        self.model = nn.Sequential(*block((opt.latent_dim + opt.n_classes), 128, normalize=False),
                                   *block(128, 256),
                                   *block(256, 512),
                                   *block(512, 1024),
                                   nn.Linear(1024, int(np.prod(img_shape))),
                                   nn.Tanh())

    def execute(self, noise, labels):
        gen_input = jt.contrib.concat((self.label_emb(labels), noise), dim=1)
        img = self.model(gen_input)
        # 将img从1024维向量变为32*32矩阵
        img = img.view((img.shape[0], *img_shape))
        return img

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)
        self.model = nn.Sequential(nn.Linear((opt.n_classes + int(np.prod(img_shape))), 512),
                                   nn.LeakyReLU(0.2),
                                   nn.Linear(512, 512),
                                   nn.Dropout(0.4),
                                   nn.LeakyReLU(0.2),
                                   nn.Linear(512, 512),
                                   nn.Dropout(0.4),
                                   nn.LeakyReLU(0.2),
                                   nn.Linear(512, 1),
                                   # TODO: 添加最后一个线性层，最终输出为一个实数
                                   )

    def execute(self, img, labels):
        d_in = jt.contrib.concat((img.view((img.shape[0], (- 1))), self.label_embedding(labels)), dim=1)
        return self.model(d_in)
        # TODO: 将d_in输入到模型中并返回计算结果

adversarial_loss = nn.MSELoss()

generator = Generator()
discriminator = Discriminator()


generator.eval()
discriminator.eval()
generator.load('./model/generator_90.pkl')
discriminator.load('./model/discriminator_90.pkl')

number ="0123456789" #TODO: 写入比赛页面中指定的数字序列（字符串类型）
n_row = len(number)
z = jt.array(np.random.normal(0, 1, (n_row, opt.latent_dim))).float32().stop_grad()
labels = jt.array(np.array([int(number[num]) for num in range(n_row)])).float32().stop_grad()
gen_imgs = generator(z,labels)

img_array = gen_imgs.data.transpose((1,2,0,3))[0].reshape((gen_imgs.shape[2], -1))
min_=img_array.min()
max_=img_array.max()
img_array=(img_array-min_)/(max_-min_)*255
Image.fromarray(np.uint8(img_array)).save("result.png")

