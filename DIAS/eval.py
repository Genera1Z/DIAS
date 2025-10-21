from pathlib import Path

import cv2
import numpy as np
import torch as pt

from object_centric_bench.datum import DataLoader
from object_centric_bench.util_datum import draw_segmentation_np
from object_centric_bench.learn import MetricWrap
from object_centric_bench.model import ModelWrap
from object_centric_bench.util import Config, build_from_config


@pt.inference_mode()
def val_epoch(cfg, dataset_v, model, loss_fn, acc_fn_v, callback_v):
    pack = Config({})
    pack.dataset_v = dataset_v
    pack.model = model
    pack.loss_fn = loss_fn
    pack.acc_fn_v = acc_fn_v
    pack.callback_v = callback_v
    pack.epoch = 0

    is_img = False  # TODO XXX

    mean = pt.from_numpy(np.array(cfg.IMAGENET_MEAN, "float32"))
    std = pt.from_numpy(np.array(cfg.IMAGENET_STD, "float32"))
    cnt = 0

    pack.isval = True
    pack.model.eval()
    [_.before_epoch(**pack) for _ in pack.callback_v]

    for i, batch in enumerate(pack.dataset_v):
        # if i < 141 or i > 141:  # TODO XXX
        #     continue
        pack.batch = {k: v.cuda() for k, v in batch.items()}

        [_.before_step(**pack) for _ in pack.callback_v]

        with pt.autocast("cuda", enabled=True):
            pack.output = pack.model(**pack)
            [_.after_forward(**pack) for _ in pack.callback_v]
            pack.loss = pack.loss_fn(**pack)
        pack.acc = pack.acc_fn_v(**pack)

        if 0:  # TODO XXX
            for image, segment in zip(pack.batch["image"], pack.output["segment2"]):
                pack.dataset_v.dataset.visualiz(
                    image, segment=segment, inorm=[mean, std], wait=0
                )

        if 0:  # TODO XXX
            # makdir
            save_dn = Path(cfg.name)
            if not Path(save_dn).exists():
                save_dn.mkdir(exist_ok=True)
            # read gt image and segment
            img_key = "image" if is_img else "video"
            imgs_gt = (  # image video
                (pack.batch[img_key] * std.cuda() + mean.cuda()).clip(0, 255).byte()
            )
            segs_gt = pack.batch["segment"].argmax(-1)  # onehot seg -> number seg
            # read pd attent -> pd segment
            if "segment2" in pack.output:
                segs_pd = pack.output["segment2"].argmax(-1)
            else:
                segs_pd = pack.output["segment"].argmax(-1)
            # visualize gt image or video
            for img_gt, seg_gt, seg_pd in zip(imgs_gt, segs_gt, segs_pd):
                if is_img:
                    img_gt, seg_gt, seg_pd = [  # warp img as vid
                        _[None] for _ in (img_gt, seg_gt, seg_pd)
                    ]
                for tcnt, (igt, sgt, spd) in enumerate(zip(img_gt, seg_gt, seg_pd)):
                    igt = cv2.cvtColor(
                        igt.permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR
                    )
                    sgt = sgt.cpu().numpy()
                    spd = spd.cpu().numpy()
                    save_path = save_dn / f"{cnt:06d}-{tcnt:06d}"
                    cv2.imwrite(f"{save_path}-i.png", igt)
                    cv2.imwrite(
                        f"{save_path}-s.png", draw_segmentation_np(igt, sgt, alpha=0.9)
                    )
                    cv2.imwrite(
                        f"{save_path}-p.png", draw_segmentation_np(igt, spd, alpha=0.9)
                    )
                cnt += 1

        [_.after_step(**pack) for _ in pack.callback_v]

    [_.after_epoch(**pack) for _ in pack.callback_v]


def main(
    # cfg_file="config-dias/dias_r-clevrtex.py",
    # cfg_file="config-dias/dias_r-coco.py",
    cfg_file="config-dias/dias_r-voc.py",
    # ckpt_file="../dias_r-clevrtex/42-0029.pth",
    # ckpt_file="../dias_r-coco/42-0027.pth",
    ckpt_file="../dias_r-voc/42-0529.pth",
):
    data_dir = "/media/GeneralZ/Storage/Static/datasets"  # TODO XXX
    pt.backends.cudnn.benchmark = True

    cfg_file = Path(cfg_file)
    data_path = Path(data_dir)
    ckpt_file = Path(ckpt_file)

    assert cfg_file.name.endswith(".py")
    assert cfg_file.is_file()
    cfg_name = cfg_file.name.split(".")[0]
    cfg = Config.fromfile(cfg_file)
    cfg.name = cfg_name

    ## datum init

    cfg.dataset_t.base_dir = cfg.dataset_v.base_dir = data_path

    dataset_v = build_from_config(cfg.dataset_v)
    dataload_v = DataLoader(
        dataset_v,
        cfg.batch_size_v // 2,  # TODO XXX  # 1,
        shuffle=False,
        num_workers=cfg.num_work,
        pin_memory=True,
    )

    ## model init

    model = build_from_config(cfg.model)
    # print(model)
    model = ModelWrap(model, cfg.model_imap, cfg.model_omap)

    if ckpt_file:
        model.load(ckpt_file, None, verbose=False)
    if cfg.freez:
        model.freez(cfg.freez, verbose=False)

    model = model.cuda()
    # model.compile()

    ## learn init

    loss_fn = MetricWrap(**build_from_config(cfg.loss_fn))
    acc_fn_v = MetricWrap(detach=True, **build_from_config(cfg.acc_fn_v))

    cfg.callback_v = [_ for _ in cfg.callback_v if _.type.__name__ != "SaveModel"]
    for cb in cfg.callback_v:
        if cb.type.__name__ == "AverageLog":
            cb.log_file = None  # TODO XXX change to current log file for eval
    callback_v = build_from_config(cfg.callback_v)

    ## do eval

    val_epoch(cfg, dataload_v, model, loss_fn, acc_fn_v, callback_v)


if __name__ == "__main__":
    main()
