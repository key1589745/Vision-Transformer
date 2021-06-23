# Vision-Transformer
Final project
### Requirements
pip install vit-pytorch
### Environment
Pytorch=1.8.1, cuda=11.2
### Run
1. Compare transformer and CNN at the same level of parameters:
```Shell
python main.py --cuda [device] --model hybrid
python main.py --cuda [device] --model CNN
```
Pre-trained models
Comparison in the same level of prameters: https://pan.baidu.com/s/1jsiaxOpWaCg9mwPM4P14Qw , extract code: 3j82
Comparison in the same level of flops: https://pan.baidu.com/s/1NmAvwvPQDScmYOHbRgy4iw , extract code: zpra

2. Compare transformer and CNN at the same level of flops:
```Shell
python main.py --cuda [device] --model CNN --net ResNet152
python main.py --cuda [device] --model hybrid --depth 6 --heads 16
```
