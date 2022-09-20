# Diversity vs. Recognizability: Human-like generalization in one-shot generative models
<img src="image/Fig1.png" width="400">

## 1. Train the one-shot generative models

### VAE-STN (VAE with spatial transformer [])
```
python train
```

### VAE-NS (Neural Statistician [])
```
python train
```

### DAGAN (Data Augmentation GAN [])
To train the version using the ResNet Architecture (i.e. DA-GAN-RN)
```
python 1_train_DAGAN.py --img_size 50 --epochs 30 --c_iter 5 --z_size 128 --out_dir DAGAN --seed 0 --architecture ResNet 
```

To train the version using the UNet Architecture (i.e. DA-GAN-UN)
```
python 1_train_DAGAN.py --img_size 50 --epochs 30 --c_iter 5 --z_size 128 --out_dir DAGAN --seed 0 --architecture UResNet 
```

## 2. Train the critic networks
### SimCLR
```
python train
```

### Protoypical Net
```
python train
```

## 3. Evaluate the one-shot generative models on the Diversity vs. Recognizability framework.

### Reference
[]
[]
[]


