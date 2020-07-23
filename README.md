# Maximum Variation Averaging
This repository contains the implementation of the so-called Maximum Variation Averaging (MVA) proposed in our paper, [Adaptive Learning Rates with Maximum Variation Averaging
](https://arxiv.org/abs/2006.11918).
MVA aims to stabilize the adaptive step size of [Adam](https://arxiv.org/abs/1412.6980)-like optimizers by adopting an adaptive weighted average of the squared gradients, where the coordinate-wise weights are chosen to maximize the estimated gradient variance. 
In this repository, we provide its implementation with [PyTorch](https://pytorch.org/) on synthetic datasets, image classification, Neural Machine Translation and Natural Language Understanding tasks, as mentioned in the experiment section of our paper. 

# Usage
We used PyTorch v1.4.0 for the experiments. 
We have divided the experiments into 3 folders:

[synthetic_data](synthetic_data): You could run [`nonconvex.py`](synthetic_data/nonconvex.py) or [`nqm.py`](synthetic_data/nqm.py) to reproduce the experiments for the nonconvex function or the Noisy Quadratic Model. 

[image_classification](image_classification): Please refer to [`launch.sh`](image_classification/launch.sh) to launch the experiments on CIFAR10 and CIFAR100. For ImageNet, you could plug the same optimizers into the [PyTorch official example code](https://github.com/pytorch/examples/blob/master/imagenet/main.py) and refer to the hyper-parameters in the paper.

[nmt_nlu](nmt_nlu): Please first enter the [`nmt_nlu`](nmt_nlu) directory and then run `pip install --editable .`. 
For Neural Machine Translation, please first [follow the steps](nmt_nlu/examples/translation#iwslt14-german-to-english-transformer) to download and process the data, and then refer to [`run-iwslt-lamadam-tristage.sh`](nmt_nlu/run-iwslt-lamadam-tristage.sh) to train a transformer with our optimizers from scratch. 
 For the GLUE benchmark, again, first [follow the steps](nmt_nlu/examples/roberta/README.glue.md) to prepare the data, download a [`RoBERTa-base`](https://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz) model and put it under `nmt_nlu/roberta-pretrained`, and use [`run-glue-base.sh`](nmt_nlu/run-glue-base.sh) to fine-tune a RoBERTa-base model on the GLUE tasks.

# Citation
Please cite as

```bibtex
@inproceedings{zhu2020mva,
  title = {Adaptive Learning Rates with Maximum Variation Averaging},
  author = {Zhu, Chen and Cheng, Yu and Gan, Zhe and Huang, Furong and Liu, Jingjing and Goldstein, Tom},
  booktitle = {arXiv: 2006.11918},
  year = {2020},
}
```
