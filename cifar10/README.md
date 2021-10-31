# Train CIFAR10 with PyTorch

The code of this example is modified from [kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar). Credits to [kuangliu](https://github.com/kuangliu).

<details>
<summary>Original README</summary>


I'm playing with [PyTorch](http://pytorch.org/) on the CIFAR10 dataset.

## Prerequisites
- Python 3.6+
- PyTorch 1.0+

## Training
```
# Start training with: 
python main.py

# You can manually resume the training with: 
python main.py --resume --lr=0.01
```

## Accuracy
| Model             | Acc.        |
| ----------------- | ----------- |
| [VGG16](https://arxiv.org/abs/1409.1556)              | 92.64%      |
| [ResNet18](https://arxiv.org/abs/1512.03385)          | 93.02%      |
| [ResNet50](https://arxiv.org/abs/1512.03385)          | 93.62%      |
| [ResNet101](https://arxiv.org/abs/1512.03385)         | 93.75%      |
| [RegNetX_200MF](https://arxiv.org/abs/2003.13678)     | 94.24%      |
| [RegNetY_400MF](https://arxiv.org/abs/2003.13678)     | 94.29%      |
| [MobileNetV2](https://arxiv.org/abs/1801.04381)       | 94.43%      |
| [ResNeXt29(32x4d)](https://arxiv.org/abs/1611.05431)  | 94.73%      |
| [ResNeXt29(2x64d)](https://arxiv.org/abs/1611.05431)  | 94.82%      |
| [SimpleDLA](https://arxiv.org/abs/1707.064)           | 94.89%      |
| [DenseNet121](https://arxiv.org/abs/1608.06993)       | 95.04%      |
| [PreActResNet18](https://arxiv.org/abs/1603.05027)    | 95.11%      |
| [DPN92](https://arxiv.org/abs/1707.01629)             | 95.16%      |
| [DLA](https://arxiv.org/pdf/1707.06484.pdf)           | 95.47%      |


</details>

<br/>

The following command can be run to train the model, where the train mode is the command line argument to specify train schedules of quantization and pruning.  We maintain all hyper parameters to be identical to the original repo except that the training epochs is set to 250. 

```bash
python3 main.py --epochs 250 --train-mode <train mode>
```

Results of different training schedules can be found in the table below. 


| Training Schedule[^1] | Train Mode |  Accuracy |
| --- | --- | --- |
| Baseline | `float` |  92.6 |
| $Q_8(w,f)$ | `quantize-late`[^2] | 92.52 |
| $P_{0.5}(w) \rightarrow Q_{8}(w, f)$ | `prune_weight-quantize` | 92.23 |
| $Q_{8}(w, f) \rightarrow  P_{0.5}(w)$ | `quantize-prune_weight-late` | 85.94 |
| $P_{0.5}(w, f) \rightarrow Q_{8}(w, f)$ | `prune_both-quantize` | 91.44 |
| $Q_{8}(w, f) \rightarrow  P_{0.5}(w, f)$ | `quantize-prune_both-late` | 86.84 |




[^1]: $P_{0.5}(w, f) \rightarrow Q_{8}(w, f)$ denotes the "prune-then-quantize" schedule on both activations and weights. The same rule applies to others.

[^2]: `quantize-late` tells the program to use a set of parameters that delay the quantization to the latter half of the training since we find early quantization hurts the accuracy to a large extend. The same rule applies to others with the `late` tag. 

## Note

Although the original repo claims to achieve 94.43% with MobileNetV2, we only achieves 92.6% with the provided hyper parameters. It is a known issue that the reported accuracy is not achieved [issues#74](https://github.com/kuangliu/pytorch-cifar/issues/74).

