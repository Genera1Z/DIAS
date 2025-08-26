from einops import rearrange
import torch.nn.functional as ptnf

from object_centric_bench.datum import (
    RandomCrop,
    Resize,
    RandomFlip,
    Normalize,
    CenterCrop,
    Lambda,
    ClevrTex,
)
from object_centric_bench.learn import (
    Adam,
    GradScaler,
    ClipGradNorm,
    CrossEntropyLoss,
    MSELoss,
    mBO,
    ARI,
    mIoU,
    CbLinearCosine,
    CbCosine,
    Callback,
    AverageLog,
    SaveModel,
)
from object_centric_bench.model import (
    DIAS,
    Sequential,
    Interpolate,
    DINO2ViT,
    Identity,
    MLP,
    NormalShared,
    SlotAttentionWithAllAttent,
    ARRandTransformerDecoder,
    LearntPositionalEmbedding,
    Linear,
    LayerNorm,
    TransformerDecoder,
    TransformerDecoderLayer,
)
from object_centric_bench.utils import Compose
from object_centric_bench.utils_model import interpolat_argmax_attent

### global

max_num = 10 + 1
resolut0 = [256, 256]
resolut1 = [16, 16]
emb_dim = 256
vfm_dim = 384

total_step = 50000  # 100000 better
val_interval = total_step // 40
batch_size_t = 64 // 2  # 64 better
batch_size_v = batch_size_t
num_work = 4
lr = 4e-4 / 2  # scale with batch_size

### datum

IMAGENET_MEAN = [[[123.675]], [[116.28]], [[103.53]]]
IMAGENET_STD = [[[58.395]], [[57.12]], [[57.375]]]
transform_t = [
    # the following 2 == RandomResizedCrop: better than max sized random crop
    dict(type=RandomCrop, keys=["image", "segment"], size=None, scale=[0.75, 1]),
    dict(type=Resize, keys=["image"], size=resolut0, interp="bilinear"),
    dict(type=Resize, keys=["segment"], size=resolut0, interp="nearest-exact", c=0),
    dict(type=RandomFlip, keys=["image", "segment"], dims=[-1], p=0.5),
    dict(type=Normalize, keys=["image"], mean=IMAGENET_MEAN, std=IMAGENET_STD),
]
transform_v = [
    dict(type=CenterCrop, keys=["image", "segment"], size=None),
    dict(type=Resize, keys=["image"], size=resolut0, interp="bilinear"),
    dict(type=Resize, keys=["segment"], size=resolut0, interp="nearest-exact", c=0),
    dict(type=Normalize, keys=["image"], mean=IMAGENET_MEAN, std=IMAGENET_STD),
]
dataset_t = dict(
    type=ClevrTex,
    data_file="clevrtex/train.lmdb",
    extra_keys=["segment"],
    transform=dict(type=Compose, transforms=transform_t),
    base_dir=...,
)
dataset_v = dict(
    type=ClevrTex,
    data_file="clevrtex/val.lmdb",
    extra_keys=["segment"],
    transform=dict(type=Compose, transforms=transform_v),
    base_dir=...,
)
collate_fn_t = None
collate_fn_v = None

### model

model = dict(
    type=DIAS,
    encode_backbone=dict(
        type=Sequential,
        modules=[
            dict(type=Interpolate, scale_factor=0.875, interp="bicubic"),
            dict(
                type=DINO2ViT,
                model_name="vit_small_patch14_reg4_dinov2.lvd142m",
                in_size=int(resolut0[0] * 0.875),
                rearrange=True,
                norm_out=False,  # >True
            ),
        ],
    ),
    encode_posit_embed=dict(type=Identity),
    encode_project=dict(
        type=MLP, in_dim=vfm_dim, dims=[vfm_dim, vfm_dim], ln="pre", dropout=0.0
    ),
    # NormalShared @42
    #   "ari": 0.2214, "ari_fg": 0.4134, "mbo": 0.3122, "miou": 0.2979
    # NormalSeparat @42
    #   "ari": 0.1963, "ari_fg": 0.4003, "mbo": 0.2945, "miou": 0.2809
    initializ=dict(type=NormalShared, num=max_num, dim=emb_dim),  # >NormalSeparat
    aggregat=dict(
        type=SlotAttentionWithAllAttent,
        num_iter=3,
        embed_dim=emb_dim,
        ffn_dim=emb_dim * 4,
        dropout=0,
        kv_dim=vfm_dim,
        trunc_bp=None,  # !!! not compatible with bi-level !!!
    ),
    decode=dict(
        type=ARRandTransformerDecoder,
        vfm_dim=vfm_dim,
        posit_embed=dict(
            type=LearntPositionalEmbedding,
            resolut=[resolut1[0] * resolut1[1]],
            embed_dim=vfm_dim,
        ),
        project1=dict(  # fc>fc+ln
            type=Sequential,
            modules=[
                dict(
                    type=Linear, in_features=vfm_dim, out_features=vfm_dim, bias=False
                ),
                dict(type=LayerNorm, normalized_shape=vfm_dim),
            ],
        ),
        project2=dict(  # fc+ln>fc
            type=Sequential,
            modules=[
                dict(
                    type=Linear, in_features=emb_dim, out_features=vfm_dim, bias=False
                ),
                dict(type=LayerNorm, normalized_shape=vfm_dim),
            ],
        ),
        backbone=dict(
            type=TransformerDecoder,
            decoder_layer=dict(
                type=TransformerDecoderLayer,
                d_model=vfm_dim,
                nhead=4,  # 4@384, 6@768
                dim_feedforward=vfm_dim * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
                bias=False,
            ),
            num_layers=4,
        ),
        readout=dict(type=Identity),
    ),
)
model_imap = dict(input="batch.image")
model_omap = ["feature", "slotz", "attent", "attent2", "recon"]
ckpt_map = []  # target<-source
freez = [r"^m\.encode_backbone\..*"]

### learn

param_groups = None
optimiz = dict(type=Adam, params=param_groups, lr=lr)
gscale = dict(type=GradScaler)
gclip = dict(type=ClipGradNorm, max_norm=1)

loss_fn = dict(
    recon=dict(
        metric=dict(type=MSELoss),
        map=dict(input="output.recon", target="output.feature"),
        transform=dict(type=Lambda, ikeys=[["target"]], func=lambda _: _.detach()),
    ),
    distill=dict(
        metric=dict(type=CrossEntropyLoss),
        map=dict(input="output.attent", target="output.attent"),
        transform=dict(
            type=Compose,
            transforms=[
                dict(type=Lambda, ikeys=[["input"]], func=lambda _: _[-3]),
                dict(type=Lambda, ikeys=[["target"]], func=lambda _: _[-1].detach()),
            ],
        ),
        weight=1,
    ),
)
_acc_dict_ = dict(
    # metric=...,
    map=dict(input="output.segment2", target="batch.segment"),
    transform=dict(
        type=Lambda,
        ikeys=[["input", "target"]],
        func=lambda _: rearrange(_, "b h w c -> b (h w) c"),
    ),
)
acc_fn_t = dict(
    mbo=dict(metric=dict(type=mBO, skip=[]), **_acc_dict_),
)
acc_fn_v = dict(
    ari=dict(metric=dict(type=ARI, skip=[]), **_acc_dict_),
    ari_fg=dict(metric=dict(type=ARI, skip=[0]), **_acc_dict_),
    mbo=dict(metric=dict(type=mBO, skip=[]), **_acc_dict_),
    miou=dict(metric=dict(type=mIoU, skip=[]), **_acc_dict_),
)

before_step = [
    dict(
        type=Lambda, ikeys=[["batch.image", "batch.segment"]], func=lambda _: _.cuda()
    ),
    dict(
        type=CbLinearCosine,
        assigns=["optimiz.param_groups[0]['lr']=value"],
        nlin=total_step // 20,
        ntotal=total_step,
        vstart=0,
        vbase=lr,
        vfinal=lr / 1e3,
    ),
    dict(
        type=CbCosine,
        assigns=["loss_fn.metrics['distill']['weight']=value"],
        ntotal=total_step,
        vbase=0,
        vfinal=1,  # Cosine0~1 > SquareWave0.1 > Constant0.1 > Constant1
    ),
]
after_forward = [
    dict(
        type=Lambda,
        ikeys=[["output.attent2"]],  # (b,s,h,w) -> (b,h,w)
        func=lambda _: interpolat_argmax_attent(_.detach(), size=resolut0),
        okeys=[["output.segment2"]],
    ),
    dict(
        type=Lambda,  # (b,h,w) -> (b,h,w,s)
        ikeys=[["output.segment2", "batch.segment"]],
        func=lambda _: ptnf.one_hot(_.long()),
    ),
]
callback_t = [
    dict(type=Callback, before_step=before_step, after_forward=after_forward),
    dict(type=AverageLog, log_file=...),
]
callback_v = [
    dict(type=Callback, before_step=before_step[:1], after_forward=after_forward),
    callback_t[1],
    dict(type=SaveModel, save_dir=..., since_step=total_step * 0.5),
]
