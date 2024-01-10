import jittor as jt
from jittor import nn, optim
from jittor.dataset.cifar import CIFAR10
from jittor.dataset.mnist import MNIST
from jittor import transform
from model import betaVAE

import os
import argparse
from tqdm import tqdm

# 用于可视化
# Windows下cuda版pytorch会与jittor冲突，可在新的conda环境中使用cpu版pytorch
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

# 开启GPU
jt.flags.use_cuda = 1


# %% 输入参数
parser = argparse.ArgumentParser(description="Generic runner for VAE models")

# 模型参数
parser.add_argument("--beta", type=float, default=0.1, help="Beta value for beta-VAE")
parser.add_argument("--latent_dim", type=int, default=64, help="Dimension of the latent space")
parser.add_argument("--load", action="store_true", help="Load saved checkpoints")
parser.add_argument("--no_save", dest="save", action="store_false", help="Save checkpoints")

# 训练参数
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer")
parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay for optimizer")
parser.add_argument("--epoch_num", type=int, default=50, help="Number of epochs to train")

# 测试参数
parser.add_argument("--test", action="store_true", help="Sample image and not train")
parser.add_argument("--sample_num", type=int, default=100, help="Number of images to sample")

# 数据集参数
parser.add_argument("--dataset", type=str, choices=["mnist", "cifar10"], default="mnist", help="Dataset to use")

# 日志参数
parser.add_argument("--log_interval", type=int, default=100, help="Interval for logging training progress")

# 路径参数
parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
parser.add_argument("--log_dir", type=str, default="log/", help="Directory to save log files")
parser.add_argument("--image_dir", type=str, default="img/", help="Directory to save generated images")

args = parser.parse_args()


# %% 设置路径
save_dir = os.path.join(args.save_dir, args.dataset)
model_save_path = os.path.join(save_dir, f"beta_{args.beta}.pkl")
optimizer_save_path = os.path.join(save_dir, f"optimizer_beta_{args.beta}.pkl")

image_dir = os.path.join(args.image_dir, args.dataset, f"beta_{args.beta}")
sample_path = os.path.join(image_dir, "sample.png")

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

writer = SummaryWriter(os.path.join(args.log_dir, args.dataset, f"beta_{args.beta}"))


# %% 加载数据
if args.dataset == "mnist":
    # MNIST
    num_channels = 1
    image_length = 28
    hidden_channels_list = [32, 64]

    # transform.ToTensor: x = x / 255, num_channels:3, Range: (0~255)->(0~1)
    # transform.Gray: x = L / 255, num_channels:1, Range: (0~255)->(0~1)
    # transform.ImageNormalize: x = (x - mean) / std
    # [mean: 0.5, std: 0.5], Range: (0~1)->(-1~1)
    # [mean: mean_x, std: std_x], data mean: 0, data std: 1
    # ImageNormalize将data均值变为0，有助于提高训练的稳定性
    transform_input = transform.Compose([transform.Gray(), transform.ImageNormalize(mean=[0.5], std=[0.5])])

    train_loader = MNIST(train=True, transform=transform_input).set_attrs(batch_size=args.batch_size, shuffle=True)
    val_loader = MNIST(train=False, transform=transform_input).set_attrs(batch_size=args.batch_size, shuffle=False)

elif args.dataset == "cifar10":
    # CIFAR10
    num_channels = 3
    image_length = 32
    hidden_channels_list = [32, 64, 128, 256]

    # transform.ToTensor: x = x / 255, num_channels:3, Range: (0~255)->(0~1), dims: [batch_size, 32, 32, 3]->[batch_size, 3, 32, 32]
    transform_input = transform.Compose([transform.ToTensor(), transform.ImageNormalize(mean=[0.5], std=[0.5])])

    train_loader = CIFAR10(train=True, transform=transform_input).set_attrs(batch_size=args.batch_size, shuffle=True)
    val_loader = CIFAR10(train=False, transform=transform_input).set_attrs(batch_size=args.batch_size, shuffle=False)

# decoder的输出范围为(-1,1)，需要转换到(0,1),以便save_image
transform_output = transform.Compose([transform.ImageNormalize(mean=[-1], std=[2])])


# %% 加载模型
model = betaVAE(num_channels, args.latent_dim, hidden_channels_list, image_length, args.beta)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

if args.load:
    model.load(model_save_path)
    optimizer.load_state_dict(jt.load(optimizer_save_path))


# %% 训练
def train(epoch):
    model.train()
    train_loss = 0

    for batch_idx, (data, _) in tqdm(enumerate(train_loader)):
        _, loss, recons_loss, kld_loss = model(data)

        optimizer.step(loss)

        train_loss += loss

        # 每隔一定batch记录一次loss
        if batch_idx % args.log_interval == 0:
            batch_num = epoch * len(train_loader) + batch_idx

            print(f"epoch: {epoch}, batch: {batch_idx}, loss: {loss.item():.6f}, recons_loss: {recons_loss.item():.6f}, kld_loss: {kld_loss.item():.6f}")
            writer.add_scalar("train_loss", loss.item(), batch_num)
            writer.add_scalar("train_recons_loss", recons_loss.item(), batch_num)
            writer.add_scalar("train_kld_loss", kld_loss.item(), batch_num)

    if args.save:
        model.save(model_save_path)
        jt.save(optimizer.state_dict(), optimizer_save_path)


# %% 测试

# 测试解码一批固定噪声的结果
fixed_noise = jt.randn(100, args.latent_dim)


# 在验证集上测试
def val(epoch):
    model.eval()
    test_loss = 0

    with jt.no_grad():
        for _, (data, _) in tqdm(enumerate(val_loader)):
            _, loss, _, _ = model(data)

            test_loss += loss

        test_loss /= len(val_loader)

        print(f"epoch: {epoch}, val loss: {test_loss.item()}")
        writer.add_scalar("val_loss", test_loss.item(), epoch)

        # 将固定噪声解码为图像
        output = model.decode(fixed_noise)

        imgs = transform_output(output).numpy()

        # make_grid: 将多张图片拼接成一张图片网格
        imgs = make_grid(torch.from_numpy(imgs), nrow=10)

        writer.add_image("Image", imgs, epoch)

        # 保存图片
        # imgs需满足 shape:[C, H, W], Range: (0~1)
        # C=1为灰度图，C=3为彩色图
        save_image(imgs, os.path.join(image_dir, f"epoch_{epoch}.png"))


# %% 采样
def sample():
    print("Sampling...")
    model.eval()

    with jt.no_grad():
        output = model.sample(args.sample_num)
        imgs = transform_output(output).numpy()
        imgs = make_grid(torch.from_numpy(imgs), nrow=10)

        save_image(imgs, sample_path)


# %% 主函数
if __name__ == "__main__":
    if args.test:
        sample()
    else:
        for epoch in range(1, args.epoch_num + 1):
            train(epoch)
            val(epoch)
