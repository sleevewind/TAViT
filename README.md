# TAViT

TAViT: Token Attention-Based Vision Transformer for Insect Pest Classification

```bash
cd TAViT
```

## Install

- Create a conda virtual environment and activate it:
```bash
conda create -n torch python=3.9 -y
conda activate torch
```

- Install pytorch 2.2.2
```bash
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```

- Install pandas
```bash
conda install pandas
```

- Install `timm==0.4.12`:
```bash
pip install timm==0.4.12
```

- Install `einops`:
```bash
pip install einops
```

- Install `Apex`:
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

If the above command is not available, please try:
python setup.py install
```

- Install other requirements:
```bash
cd ..
pip install opencv-python==4.9.0.80 termcolor==2.4.0 yacs==0.1.8
```

## Data preparation

### Pest120

This dataset will be publicly available soon.

The file structure should look like:

```bash
$ tree data
  Pest120
  ├── train
  |   ├── class1
  |   │   ├── img1.jpg
  |   │   ├── img2.jpg
  |   │   ├── img3.jpg
  |   │   └── ...
  |   ├── class2
  |   │   ├── img1.jpg
  |   │   ├── img2.jpg
  |   │   ├── img3.jpg
  |   │   └── ...
  |   └── ...
  ├── test
  |   └─ ...
  ├── val 
  |   └─ ...
  ├── classes.csv
  ├── train.csv
  ├── test.csv
  └── val.csv
```

### IP102

This dataset can be obtained at: [IP102-Dataset (kaggle.com)](https://www.kaggle.com/datasets/rtlmhjbn/ip02-dataset)

The file structure should look like:

```bash
$ tree data
  IP102
  ├── classification
  |   ├── train
  │   |   ├── class1
  │   |   │   ├── img1.jpg
  │   |   │   ├── img2.jpg
  │   |   │   ├── img3.jpg
  │   |   │   └── ...
  │   |   ├── class2
  │   |   │   ├── img1.jpg
  │   |   │   ├── img2.jpg
  │   |   │   ├── img3.jpg
  │   |   │   └── ...
  │   |   └── ...
  |   ├── test
  |   └── val
  ├── classes.txt
  ├── test.txt
  ├── train.txt
  └── val.txt
```

### CPB

This dataset can be obtained at: [edsonbollis/Citrus-Pest-Benchmark: Citrus Pest Benchmark (github.com)](https://github.com/edsonbollis/Citrus-Pest-Benchmark)

The file structure should look like:

```bash
  $ tree data
  CPB
  ├── img1.jpg
  ├── img2.jpg
  ├── img3.jpg
  ├── ...
  ├── pests_test_original.csv
  ├── pests_train_no_blured_manual.csv
  ├── pests_train_original.csv
  ├── pests_validation_no_blured_manual.csv
  └── pests_validation_original.csv
```

## Evaluation

To evaluate a pre-trained `TAViT` on **Pest120** val, run:
```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> --master_port 12345 main.py --eval 
--cfg <config-file> --resume <checkpoint> --dataset Pest120 --data-path <data-path> 
```

For example, to evaluate the `TAViT` with a single GPU:
```bash
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py --eval --cfg configs/Pest120/TAViT.yaml --resume ckpt_epoch_xxx.pth  --dataset Pest120 --data-path data/Datasets/Pest120
```

## Training from scratch

To train a `TAViT` on **Pest120** from scratch, run:
```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> --master_port 12345  main.py \ 
--cfg <config-file> --dataset Pest120 --data-path <data-path> \
[--batch-size <batch-size-per-gpu> --output <output-directory> --tag <job-tag>]
```

+ Specifically, to train `TAViT` with 2 GPU of a single node on **Pest120** for 300 epochs, run:
```bash
python -m torch.distributed.launch --nproc_per_node 2 --master_port 12345  main.py --cfg configs/Pest120/TAViT.yaml --dataset Pest120 --data-path data/Datasets/Pest120 --batch-size 128
```
+ To train `TAViT` with 2 GPU of a single node on **IP102** for 300 epochs, run:
```bash
python -m torch.distributed.launch --nproc_per_node 2 --master_port 12345  main.py --cfg configs/IP102/TAViT.yaml --dataset IP102 --data-path data/Datasets/IP102/classification/ --batch-size 128
```
+ To train `TAViT` with 2 GPU of a single node on **CPB** for 300 epochs, run:
```bash
 python -m torch.distributed.launch --nproc_per_node 2 --master_port 12345  main.py --cfg configs/CPB/TAViT.yaml --dataset CPB --data-path data/Datasets/CPB/ --batch-size 128
```
## Throughput

To measure the throughput, run:
```bash
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py --cfg <config-file> --dataset <dataset> --data-path <data-path> --batch-size 64 --throughput --amp-opt-level O0
```
For example, to evaluate the throughput of `TAVIT` with a single GPU:

```bash
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py --cfg configs/Pest120/TAViT.yaml --dataset Pest120 --data-path ../research/data/Datasets/Pest120 --batch-size 64 --throughput --amp-opt-level O0
```

