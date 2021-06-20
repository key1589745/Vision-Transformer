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

2. Compare transformer and CNN at the same level of flops:
