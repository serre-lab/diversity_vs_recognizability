# Diversity vs. Recognizability: Human-like generalization in one-shot generative models
<img src="image/Fig1.png" width="400"><img src="image/Fig1.png" width="400">

Before training any generative model, you need to download the [Omniglot dataset](https://github.com/brendenlake/omniglot) [1] (the images_background.zip and images_evaluation files for python). Then unzip those files and rename the folders 'background' and 'evaluation', respectively. In the command line below, you need to specify the dataset path in `--dataset_root`.

## 1. Train the one-shot generative models

### VAE-STN (VAE with spatial transformer [2])
```
python3 1_train_vaestn.py --device cuda:0 --z_size 80 --time_step 100 --out_dir outdir --dataset_rout dataset_root --beta 1
```
Do not forget to change `--out_dir` and `--dataset_root` args to your saving path and data path (omniglot). If you want to reproduce all the VAE-STN models presented in Fig. 3a (light blue data points), you need to run this command line and vary `--beta` from 0 to 4 (step 0.25),   `--time_step` from 20 to 90 (step 10) and `--z_size` from 10 to 400 (step 25).

### VAE-NS (Neural Statistician [3])
```
python3 1_train_NS.py --device cuda:0 --model_name ns --sample-size 5 --z-dim 16 --exemplar --epoch 300 --learning_rate 1e-3 --beta 1 --out_dir outdir --dataset_root dataset_root
    
```
Do not forget to change `--out_dir` and `--dataset_root` args to your saving path and data path (omniglot). If you want ot reproduce all the VAE-NS models presented in Fig3. a, run this command line and vary `--sample-size` from 2 to 20 (step 1), `--beta` from 0 to 5 (step 0.25) and `--z_size` from 0 to 100 (step 10)

### DAGAN (Data Augmentation GAN [4])
To train the version using the ResNet Architecture (i.e. DA-GAN-RN)

```
python3 1_train_DAGAN.py --epochs 30 --device cuda:0 --c_iter 5 --z_size 128 --out_dir out_dir --dataset_root dataset_root --architecture ResNet 
```

To train the version using the UNet Architecture (i.e. DA-GAN-UN)
```
python3 1_train_DAGAN.py --epochs 30 --device cuda:0 --c_iter 5 --z_size 128 --out_dir out_dir --dataset_root dataset_root --architecture UResNet 
```

Do not forget to change `--out_dir` and `--dataset_root` args to your saving path and data path (omniglot). If you want to reproduce all the DA-GAN models presented in Fig. 3a, you need to run these command lines and vary `--z_size` from 10 to 1000 (by step of 10).

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


