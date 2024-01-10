# Jittor—beta-VAE

清华大学“媒体计算”课程大作业，基于jittor实现beta-VAE

## Reference

### 理论

- [变分自编码器（一）：原来是这么一回事 - 科学空间|Scientific Spaces](https://spaces.ac.cn/archives/5253)
- [令人拍案叫绝的Wasserstein GAN - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/25071913)

### 代码

- https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py

## Usage

### 训练

```bash
python main.py --dataset mnist --epoch 100 --learning_rate 1e-4 --batch_size 64
```

更多参数请参考`main.py`

### 测试

```bash
python main.py --test
```

### Tensorboard

```bash
tensorboard --logdir=log
```
