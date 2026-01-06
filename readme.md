# `DIAS` Slot Attention with Re-Initialization and Self-Distillation



<br>
<br>

## ⚗️ (2026/01/06) Update !!!

Please check our brand new OCL works:
- **[RandSF.Q](https://github.com/Genera1Z/RandSF.Q)**: significantly surpasses state-of-the-art video OCL, e.g., **SlotContrast**, by **up to 10 points**!
- **[SmoothSA](https://github.com/Genera1Z/SmoothSA)**: improves the state of the art **even further**, e.g., **SPOT** / **DIAS** (images) and **SlotContrast** / **RandSF.Q** (videos), with **minimal modifications**!

<br>
<br>
<br>

---



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

⭐⭐⭐ ***Please check GitHub repo [VQ-VFM-OCL](https://github.com/Genera1Z/VQ-VFM-OCL).*** ⭐⭐⭐



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

Take DIAS on COCO as an example.

**(1) Environment**

To set up the environment, run:
```shell
# python 3.11
pip install -r requirements.txt
```

**(2) Dataset**

To prepare the dataset, download ***Converted Datasets*** and unzip to `path/to/your/dataset/`. Or convert them by yourself according to ```XxxDataset.convert_dataset()``` docs.

**(3) Train**

To train the model, run:
```shell
python train.py \
    --seed 42 \
    --cfg_file config-dias/dias_r-coco.py \
    --data_dir path/to/your/dataset \
    --save_dir save
```

**(4) Evaluate**

To evaluate the model, run:
```shell
python eval.py \
    --cfg_file config-dias/dias_r-coco.py \
    --data_dir path/to/your/dataset \
    --ckpt_file archive-dias/dias_r-coco/best.pth \
    --is_viz True \
    --is_img True
# object discovery accuracy values will be printed in the terminal
# object discovery visualization will be saved to ./dias_r-coco/
```



## 🤗 Contact & Support

If you have any issues on this repo or cool ideas on OCL, please do not hesitate to contact me!
- page: https://genera1z.github.io
- email: rongzhen.zhao@aalto.fi, zhaorongzhenagi@gmail.com

If you are applying OCL (not limited to this repo) to tasks like **visual question answering**, **visual prediction/reasoning**, **world modeling** and **reinforcement learning**, let us collaborate!



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
