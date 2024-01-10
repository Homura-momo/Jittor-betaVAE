"""
VAE——Jittor
ref: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
"""

import jittor as jt
from jittor import nn


class betaVAE(nn.Module):
    def __init__(self, in_channels, latent_dim, hidden_channels_list, image_length, beta):
        """
        :latent_dim: 隐向量Z的维度
        :hidden_channels_list: encoder各个卷积层的通道数
        :image_length: 输入图像的边长
        """
        super().__init__()

        self.latent_dim = latent_dim
        cur_channels = in_channels

        # beta为超参，用于控制KL散度的权重
        self.beta = beta

        # encoder，将输入图像编码为隐向量分布的均值和方差
        self.encoder = nn.Sequential()

        # encoder的卷积层
        for hidden_channels in hidden_channels_list:
            # 卷积，将图像缩小
            # 卷积后的图像大小计算公式为：N=(W−F+2P)//S​+1
            # 其中 W 表示输入图像的大小，F 表示卷积核的大小，S 表示步长，P 表示填充的像素数
            self.encoder.append(nn.Conv2d(cur_channels, hidden_channels, kernel_size=3, stride=2, padding=1))
            self.encoder.append(nn.BatchNorm2d(hidden_channels))
            self.encoder.append(nn.LeakyReLU())

            # 若F=3，S=2，P=1，则卷积后的图像大小为原图像大小的一半
            image_length //= 2
            # in_channels更新为上一层的输出通道数
            cur_channels = hidden_channels

        # encoder的全连接层
        # 输入为encoder卷积层的输出，展平后长度为 cur_channels * image_size**2
        # 输出为隐向量分布的均值和方差，长度为 latent_dim
        self.fc_mu = nn.Linear(cur_channels * image_length**2, latent_dim)
        self.fc_var = nn.Linear(cur_channels * image_length**2, latent_dim)

        # 将隐向量从latent_dim映射到更大维度，之后再reshape为多个通道的特征图
        self.decoder_projection = nn.Linear(latent_dim, cur_channels * image_length**2)

        # 输入decoder的图像的[通道数,图像高，图像宽]
        self.decoder_input_chw = (cur_channels, image_length, image_length)

        # decoder，将隐向量解码为图像
        self.decoder = nn.Sequential()

        # decoder的卷积层
        # 各层通道数顺序与encoder相反
        for i in range(len(hidden_channels_list) - 1, 0, -1):
            # 转置卷积，将图像放大
            # 转置卷积后的图像大小计算公式为：N=(W−1)∗S−2P+F+Pout
            # 其中 W 表示输入图像的大小，F 表示卷积核的大小，S 表示步长，P 表示填充的像素数，Pout 表示输出填充的像素数
            self.decoder.append(nn.ConvTranspose2d(hidden_channels_list[i], hidden_channels_list[i - 1], kernel_size=3, stride=2, padding=1, output_padding=1))
            self.decoder.append(nn.BatchNorm2d(hidden_channels_list[i - 1]))
            self.decoder.append(nn.LeakyReLU())

            # 若F=3，S=2，P=1，Pout=1，则转置卷积后的图像大小为原图像大小的两倍
            image_length *= 2

        # decoder的最后一层，将通道数变为输入图像的通道数
        # 最后的激活函数使用tanh，将输出值限制在[-1,1]之间
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels_list[0], hidden_channels_list[0], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_channels_list[0]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_channels_list[0], in_channels, kernel_size=3, padding=1),
            nn.Tanh(), 
        )

    def encode(self, input):
        """
        编码过程，图像->隐向量分布
        :param input: (Var) [B x H x W x C]
        :return: (Tuple) (mu, log_var) ([B x D], [B x D])
        """
        result = self.encoder(input)

        # 将多个通道的特征图展平为一维向量
        # [B x C x H x W] -> [B x C*H*W]
        result = jt.flatten(result, start_dim=1)

        # 映射为高斯分布的均值和方差
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return mu, log_var

    def decode(self, z):
        """
        解码过程，隐向量->图像
        :param z: (Var) [B x D]
        :return: (Var) [B x C x H x W] Range: [-1, 1]
        """
        # 隐向量扩充维度
        result = self.decoder_projection(z)

        # 从一维向量变形为多个通道的特征图
        result = result.view(-1, *self.decoder_input_chw)

        # 放大为图像
        result = self.decoder(result)
        result = self.final_layer(result)

        return result

    def reparameterize(self, mu, logvar):
        """
        重参数化，隐向量分布->隐向量
        :相当于给latent code加入噪声
        :ref: https://spaces.ac.cn/archives/5253
        :param mu: (Var) 高斯分布的均值 [B x D]
        :param logvar: (Var) 高斯分布的方差的自然对数 [B x D]
        :return: (Var) [B x D]
        :z ~ N(mu, std)
        :eps ~ N(0, 1)
        :z = mu + std * eps
        """
        # 计算得到标准差
        std = jt.exp(logvar / 2)
        # 从标准正态分布N(0, 1)中采样得到eps
        eps = jt.randn_like(mu)

        return eps * std + mu

    def execute(self, input):
        """
        :param input: (Var) [B x C x H x W] Range: [-1, 1]
        :return: (Tuple) (recons,loss)
        :return recons: (Var) [B x C x H x W], 重建的图像 Range: [-1, 1]
        :return loss,recons_loss,kld_loss: 本轮batch的loss
        """

        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        recons = self.decode(z)
        loss, recons_loss, kld_loss = self.loss_fn(input, recons, mu, log_var)

        return recons, loss, recons_loss, kld_loss

    def loss_fn(self, input, recons, mu, log_var):
        """
        计算VAE的loss，包括重建误差和KL散度
        :param input: 输入图像
        :param recons: Z经过decoder重建生成的图像
        :param mu: 隐向量分布的均值
        :param log_var: 隐向量分布的方差的自然对数
        :return recons_loss: 重建误差
        :return kld_loss: KL散度
        :return loss: recons_loss + self.beta * kld_loss
        """
        # 重建误差
        recons_loss = nn.mse_loss(recons, input)

        # P(Z|X) 和 N(0,1) 的KL散度
        # 使P(Z|X)的分布尽量接近N(0,1)
        kld_loss = jt.mean(0.5 * (mu**2 + jt.exp(log_var) - log_var - 1))

        loss = recons_loss + self.beta * kld_loss

        return loss, recons_loss, kld_loss

    def sample(self, num_samples):
        """
        从N(0, 1)中采样得到隐向量，然后解码为图像
        :param num_samples: (Int) Number of samples
        :return: (Var) [B x C x H x W] Range: [-1, 1]
        """
        z = jt.randn(num_samples, self.latent_dim)

        result = self.decode(z)
        return result
