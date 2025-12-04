# `DIAS` Slot Attention with Re-Initialization and Self-Distillation



[![](https://img.shields.io/badge/arXiv-2507.23755-red)](https://arxiv.org/abs/2507.23755)
[![](https://img.shields.io/badge/license-MIT-orange)](LICENSE)
[![](https://img.shields.io/badge/python-3.11-yellow)](https://www.python.org)
[![](https://img.shields.io/badge/pytorch-2.6-green)](https://pytorch.org)
[![](https://img.shields.io/badge/model-checkpoints-blue)](https://github.com/Genera1Z/DIAS?tab=readme-ov-file#-model-checkpoints--training-logs)
[![](https://img.shields.io/badge/training-logs-purple)](https://github.com/Genera1Z/DIAS?tab=readme-ov-file#-model-checkpoints--training-logs)



Unlike popular solutions based on dense feature maps, Object-Centric Learning (OCL) represents visual scenes as sub-symbolic object-level feature vectors, termed slots, which are highly versatile for tasks involving visual modalities. OCL typically aggregates object superpixels into slots by iteratively applying competitive cross attention, known as Slot Attention, with the slots as the query. However, once initialized, these slots are reused naively, causing redundant slots to compete with informative ones for representing objects. This often results in objects being erroneously segmented into parts. Additionally, mainstream methods derive supervision signals solely from decoding slots into the input's reconstruction, overlooking potential supervision based on internal information. To address these issues, we propose Slot Attention with re-Initialization and self-Distillation (DIAS): $\emph{i)}$ We reduce redundancy in the aggregated slots and re-initialize extra aggregation to update the remaining slots; $\emph{ii)}$ We drive the bad attention map at the first aggregation iteration to approximate the good at the last iteration to enable self-distillation. Experiments demonstrate that DIAS achieves state-of-the-art on OCL tasks like object discovery and recognition, while also improving advanced visual prediction and reasoning.



## 🎉 Accepted to ACM MM 2025 as a Poster

Official implementation of ACM MM 2025 paper "**Slot Attention with Re-Initialization and Self-Distillation**".
Please note that features *slot pruning*, along with *re-initialization*, are not included.

## 🏆 Performance

### (1) ⭐⭐⭐ Re-evaluated Performance Values @ Version 3 ⭐⭐⭐

**Object discovery**:

|                 |    ari   |   arifg  |    mbo   |   miou   |
|-----------------|:--------:|:--------:|:--------:|:--------:|
| dias_r-clevrtex | 80.9±0.3 | 79.1±0.3 | 63.3±0.1 | 61.9±0.0 |
| dias_r-coco     | 25.6±0.1 | 41.2±0.3 | 31.7±0.1 | 30.2±0.1 |
| dias_r-voc      | 30.9±0.5 | 33.5±0.7 | 43.4±0.5 | 42.4±0.5 |


### (2) Old Performance Values

**Object discovery performance**.

| DIAS @ DINO2-S/14, 256x256 (224)    |    ARI   |   ARIfg  |    mBO   |   mIoU   |
|:------------------|:--------:|:--------:|:--------:|:--------:|
| CLEVRTEX #slot=11 | 80.9±0.3 | 79.1±0.3 | 63.3±0.0 | 61.9±0.0 |
| MS COCO #slot=7   | 22.0±0.2 | 41.4±0.2 | 31.1±0.1 | 29.7±0.1 |
| Pascal VOC #slot=6| 26.6±1.0 | 33.7±1.5 | 43.3±0.3 | 42.4±0.3 |

For my implementation of baseline methods and their model checkpoints, please visit my repo [VQ-VFM-OCL](https://github.com/Genera1Z/VQ-VFM-OCL).



## 🌟 Highlights

⭐⭐⭐ ***Inherited from GitHub repo [VQ-VFM-OCL](https://github.com/Genera1Z/VQ-VFM-OCL).*** ⭐⭐⭐

- ✅ **fp16 fast training** [Automatic mixed precision](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html) training (fp32+fp16) is enabled. Most of the training can be finished less than 4 or 8 hours (for image or video OCL respectively) using one V100 GPU.
- ✅ **less I/O overhead** Datasets are stored in [LMBD](https://lmdb.readthedocs.io) database format to save I/O overhead, beneficial especially on computing cluster.

- ✅ **config-driven experiment** This is totally config-driven framework, largely inspired by [OpenMMLab](https://github.com/open-mmlab), but with much less capsulation.

- ✅ **strong baselines** <!-- All models requiring VAE are implemented with StableDiffusion pretrained VAE [TinyVAE](https://huggingface.co/docs/diffusers/v0.30.1/en/api/models/autoencoder_tiny); --> All models are trained with [strong](https://arxiv.org/abs/2206.07764) data augmentations; All models employ vision foundation model [DINO2](https://huggingface.co/docs/transformers/en/model_doc/dinov2) as their backbone.



## 🚑️ Changelogs

- [2025/11/07] Fix ``lmdb`` multiprocessing issues due to ``torch>=3.7``.
- ⭐⭐⭐ [2025/10/21] ⭐⭐⭐ **Object discovery accuracy values are updated for version 3. Check this table file [acc-v3.xlsx](acc-v3.xlsx) for details**.
- [2025/10/20] Version 3 (based on [VQ-VFM-OCL](https://github.com/Genera1Z/VQ-VFM-OCL)): re-implement segmentation evaluation; corresponding new dataset lmdb files are uploaded. Thus, object discovery acc could change a little, especially ARI values.



## 🧭 Repo Stucture

[Source code](https://github.com/Genera1Z/DIAS).
```shell
- config-dias/          # *** configs for our DIAS ***
- object_centric_bench/
  - datum/              # dataset loading and preprocessing
  - model/              # model building
    - ...
    - dias.py           # *** for our DIAS model building ***
    - ...
  - learn/              # metrics, optimizers and callbacks
- convert.py
- train.py
- eval.py
- requirements.txt
```

[Releases](https://github.com/Genera1Z/DIAS/releases).
```shell
- archive-dias/      # our DIAS models and logs
```



## 🚀 Converted Datasets

Datasets ClevrTex, COCO and VOC, which are converted into LMDB format and can be used off-the-shelf, are available as [releases](https://github.com/Genera1Z/VQ-VFM-OCL/releases).
- [dataset-clevrtex](https://github.com/Genera1Z/VQ-VFM-OCL/releases/tag/dataset-clevrtex): converted dataset [ClevrTex](https://www.robots.ox.ac.uk/~vgg/data/clevrtex).
- [dataset-coco](https://github.com/Genera1Z/VQ-VFM-OCL/releases/tag/dataset-coco): converted dataset [COCO](https://cocodataset.org).
- [dataset-voc](https://github.com/Genera1Z/VQ-VFM-OCL/releases/tag/dataset-voc): converted dataset [VOC](http://host.robots.ox.ac.uk/pascal/VOC).



## 🧠 Model Checkpoints & Training Logs

**The checkpoints and training logs (@ random seeds 42, 43 and 44) for all models in the table above** are available as [releases](https://github.com/Genera1Z/DIAS/releases).
- [archive-dias](https://github.com/Genera1Z/DIAS/releases/tag/archive-dias): model checkpoints and train/val logs of DIAS trained on datasets CLEVRTEX, Microsoft COCO and Pascal VOC.



## 🔥 How to Use


### (1) Install

(Using Python version 3.11)
```shell
pip install -r requirements.txt
```
Use package versions no older than the specification.


### (2) Prepare

Download **converted datasets** or convert original datasets into LMDB format: 
```shell
python convert.py
```
But **firstly** download original datasets according to docs of ```XxxDataset.convert_dataset()```.


### (3) Train

Run training:
```shell
python train.py
```
But **firstly** change the arguments marked with ```TODO XXX``` to your needs.

For example,
```shell
python train.py \
    --seed 42 \
    --cfg_file config-dias/dias_r-coco.py \
    --data_dir path/to/coco
```


### (4) Evaluate

Run evaluation:
```shell
python eval.py
```
But **firstly** modify places marked with ``TODO XXX`` according to your needs.



## 💡 Tips

1. Any config file can be converted into typical Python code by changing from
```Python
model = dict(type=ClassName, key1=value1,..)
```
to
```Python
model = ClassName(key1=value1,..)
```

2. All config files follow a similar structure, and you can use file comparator [Meld](https://meldmerge.org) with [VSCode](https://code.visualstudio.com/) plugin [Meld Diff](https://marketplace.visualstudio.com/items?itemName=danielroedl.meld-diff) to check their differences.

<img src="https://github.com/Genera1Z/VQ-VFM-OCL/raw/main/res/meld_diff.png" style="width:100%;">



## 🤗 Contact & Support

I am now working on Object-Centric Learning (OCL). If you have any cool ideas or issues, do not hasitate to contact me!
- Website: [genera1z.github.io](https://genera1z.github.io)
- WeChat: Genera1Z
- GoogleScholar: [MqlwrKAAAAAJ](https://scholar.google.com/citations?hl=en&user=MqlwrKAAAAAJ&view_op=list_works&sortby=pubdate)
- LinkedIn: [rongzhen-zhao-3b7215247](https://www.linkedin.com/in/rongzhen-zhao-3b7215247)
- eMail: rongzhen.zhao@aalto.fi, zhaorongzhenagi@gmail.com

If you are applying OCL (not limited to this repo) to tasks like **visual question answering**, **visual prediction/reasoning**, **world modeling** and **reinforcement learning**, let us collaborate!



## ⚗️ Further Research

My further research works on OCL can be found in [my repos](https://github.com/Genera1Z?tab=repositories) or [my academic page](https://genera1z.github.io).



## 📚 Citation

If you find this repo useful, please cite our work.
```
@article{zhao2025dias,
  title={{Slot Attention with Re-Initialization and Self-Distillation}},
  author={Zhao, Rongzhen and Zhao, Yi and Kannala, Juho and Pajarinen, Joni},
  journal={ACM MM},
  year={2025}
}
```
