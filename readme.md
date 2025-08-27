# DIAS : Slot Attention with Re-Initialization and Self-Distillation



## About

Official implementation of ACM MM 2025 paper "**Slot Attention with Re-Initialization and Self-Distillation**" available on [arXiv:2507.23755](https://arxiv.org/abs/2507.23755).
Please note that slot pruning (along with re-initialization) is not implemented.

| DIAS @ DINO2-S/14, 256x256 (224)    |    ARI   |   ARIfg  |    mBO   |   mIoU   |
|:------------------|:--------:|:--------:|:--------:|:--------:|
| CLEVRTEX #slot=11 | 80.9±0.3 | 79.1±0.3 | 63.3±0.0 | 61.9±0.0 |
| MS COCO #slot=7   | 22.0±0.2 | 41.4±0.2 | 31.1±0.1 | 29.7±0.1 |
| Pascal VOC #slot=6| 26.6±1.0 | 33.7±1.5 | 43.3±0.3 | 42.4±0.3 |

For my implementation of baseline methods, please visit the repos as below: [TODO]
- Auto-regressive decoding: [SLATE](https://github.com/singhgautam/slate) vs VVO-Tfd, [STEVE](https://github.com/singhgautam/steve) vs VVO-TfdT, [SPOT](https://github.com/gkakogeorgiou/spot) vs VVO-Tfd9
- Mixture-based decoding: [DINOSAUR](https://github.com/martius-lab/videosaur) vs VVO-Mlp, [VideSAUR](https://github.com/martius-lab/videosaur) vs VVO-SmdT
- Diffusion-based decoding: [SlotDiffusion](https://github.com/Wuziyi616/SlotDiffusion) vs VVO-Dfz



## Stucture

```
- config-dias  # configs for DIAS
    └ *.py
- object_centric_bench
    └ datum  # implementations of datasets ClevrTex, COCO, VOC
        └ *.py
    └ model  # modules that compose OCL models
        └ *.py
    └ learn  # callbacks, metrics and optimizers
        └ *.py
    └ *.py
- convert.py
- train.py
- eval.py
- requirements.txt
```

**Core code for paper DIAS**: 
- ``object_centric_bench/model/dias.py``;



## Converted Datasets 🚀

Converted datasets, including ClevrTex, COCO, VOC and MOVi-D are available as [releases](https://github.com/Genera1Z/DIAS/releases).
- [dataset-clevrtex](https://github.com/Genera1Z/VQ-VFM-OCL/releases/tag/dataset-clevrtex): converted dataset [ClevrTex](https://www.robots.ox.ac.uk/~vgg/data/clevrtex).
- [dataset-coco](https://github.com/Genera1Z/VQ-VFM-OCL/releases/tag/dataset-coco): converted dataset [COCO](https://cocodataset.org). [TODO upload new version]
- [dataset-voc](https://github.com/Genera1Z/VQ-VFM-OCL/releases/tag/dataset-voc): converted dataset [VOC](http://host.robots.ox.ac.uk/pascal/VOC).



## Model Checkpoints 🌟

***The checkpoints for all the models in the two tables above*** are available as [releases](https://github.com/Genera1Z/DIAS/releases).
- [dias_r-clevrtex](https://github.com/Genera1Z/DIAS/releases/tag/dias_r-clevrtex): model checkpoints and train/val logs of DIAS trained on datasets CLEVRTEX, under three random seeds 42, 43 and 44.
- [dias_r-coco](https://github.com/Genera1Z/DIAS/releases/tag/dias_r-coco): model checkpoints and train/val logs of DIAS trained on datasets MS COCO, under three random seeds 42, 43 and 44.
- [dias_r-voc](https://github.com/Genera1Z/DIAS/releases/tag/dias_r-voc): model checkpoints and train/val logs of DIAS trained on datasets Pascal VOC, under three random seeds 42, 43 and 44.



## How to Use [TODO update]

#### (1) Install requirements

(Using Python version 3.11)
```shell
pip install -r requirements.txt
```
Use package versions no older than the specification.

#### (2) Prepare datasets

You can download the converted LMDB-format datasets ... [TODO]. 
Or by yourself convert original datasets into LMDB format: 
```shell
python convert.py
```
But **firstly** download original datasets according to docs of ```XxxDataset.convert_dataset()```.

#### (3) Pretrain and train

Run training:
```shell
python train.py
```
But **firstly** change the arguments marked with ```TODO XXX``` to your needs.

Specifically on training:
- For SLATE/STEVE, SlotDiffusion and VQDINO-Tfd/Mlp/Dfz, there are two stages for training. For example,
```shell
# 1. pretrain the VAE module
python train.py --cfg_file config-slatesteve/vqvae-coco-c256.py
# *. place the best VAE checkpoint at archive-slatesteve/vqvae-coco-c256/best.pth
mv save archive-slatesteve
# 2. train the OCL model
python train.py --cfg_file config-slatesteve/slate_r_vqvae-coco.py --ckpt_file archive-slatesteve/vqvae-coco-c256/best.pth
```
 - VQDINO-Tfd/Mlp models share the same ``config-vqdino/vqdino-xxx-c256.py`` and corresponding checkpoint as VAE pretraining;
 - VQDINO-Dfz models take ``config-vqdino/vqdino-xxx-c4.py`` and corresponding checkpoint as VAE pretraining.

- For DINOSAUR, there is only one training stage. For example,
```shell
python train.py --cfg-file config-dinosaur/dinosaur_r-coco.py
```

#### (4) Evaluate

Run evaluation:
```shell
python eval.py
```
Remember **firstly** modify the script according to your need.



## Tips [TODO update -> about the framework, the philosophy]

1. Any config file can be converted into typical Python code by changing from
```Python
...
model = dict(type="class_name", key1=value1,..)
...
```
to
```Python
from object_centric_bench.datum import *
from object_centric_bench.model import *
from object_centric_bench.learn import *
...
model = class_name(key1=value1,..)
...
```

2. All config files follow a similar structure, and you can use file comparator [Meld](https://meldmerge.org) with VSCode plugin [Meld Diff](https://marketplace.visualstudio.com/items?itemName=danielroedl.meld-diff) to check their differences.
<img src="res/meld_diff.png" style="width:75%;">



## About Me 🤗

I am now working on object-centric learning (OCL). If you have any cool ideas on OCL or issues about this repo, just contact me.
- WeChat: Genera1Z
- email: rongzhen.zhao@aalto.fi, zhaorongzhenagi@gmail.com

If you are **applying OCL (not limited to this repo) to tasks like visual question answering, visual prediction/reasoning, world modeling and reinforcement learning**, I am also willing to be of your help.



## Citation

If you find this repo useful, please cite our work.
```
@article{zhao2025dias,
  title={{Slot Attention with Re-Initialization and Self-Distillation}},
  author={Zhao, Rongzhen and Zhao, Yi and Kannala, Juho and Pajarinen, Joni},
  journal={ACM MM},
  year={2025}
}
```
