import numpy as np

from detector import *

def _box_iou_xyxy_1v1(a, b, eps=1e-9):
    # a, b: (4,) xyxy in pixels
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / (union + eps)


def print_letterbox_params_compare(*, H0: int, W0: int, new_h: int, new_w: int, stride: int = 32):
    # Compare Ultralytics-style LetterBox(auto=False) vs "auto pad to stride" (auto=True)
    for auto in (False, True):
        r, left, top, new_unpad_w, new_unpad_h = letterbox_params_ultra(
            H0, W0, new_h, new_w, stride=stride, auto=auto
        )

        # What the final padded canvas size would be if we mirrored Ultralytics' rounding:
        # top = round(dh-0.1), bottom = round(dh+0.1) with dh = (pad_h/2) when center=True
        # Same for left/right.
        # Here we approximate by using symmetric padding:
        final_w_approx = new_unpad_w + 2 * left
        final_h_approx = new_unpad_h + 2 * top

        print(
            f"auto={auto} r={r:.6f} left={left} top={top} "
            f"new_unpad=(w={new_unpad_w}, h={new_unpad_h}) "
            f"final_approx=(w={final_w_approx}, h={final_h_approx}) "
            f"target=(w={new_w}, h={new_h})"
        )


def roundtrip_letterbox_one_box(
    *,
    dataset,
    sample,                      # a FiftyOne sample
    det,                         # a detection dict/object with ["bounding_box"]
    new_h: int,
    new_w: int,
):
    # Ensure metadata available
    if sample.metadata is None:
        dataset.select([sample.id]).compute_metadata()
        sample = dataset[sample.id]

    W0, H0 = int(sample.metadata.width), int(sample.metadata.height)

    print_letterbox_params_compare(H0=H0, W0=W0, new_h=new_h, new_w=new_w, stride=32)

    # GT box: original normalized xywh -> original pixels xyxy
    gt_xywh_norm = np.array(det["bounding_box"], dtype=np.float32)
    gt_xyxy_orig = xywh_norm_to_xyxy_pix(gt_xywh_norm, W0, H0)

    # orig -> letterbox (pixels)
    gt_xyxy_lb = map_box_orig_to_letterbox_xyxy_pix(gt_xyxy_orig, H0, W0, new_h, new_w)

    # letterbox -> orig (pixels)
    rt_xyxy_orig = map_box_letterbox_to_orig_xyxy_pix(gt_xyxy_lb, H0, W0, new_h, new_w)

    # Compare
    abs_err = np.abs(rt_xyxy_orig - gt_xyxy_orig)
    max_abs_err = float(abs_err.max())
    iou = float(_box_iou_xyxy_1v1(gt_xyxy_orig, rt_xyxy_orig))

    print("orig xyxy:", gt_xyxy_orig)
    print("rt   xyxy:", rt_xyxy_orig)
    print("abs err  :", abs_err, "max_abs_err(px)=", max_abs_err)
    print("roundtrip IoU:", iou)

    return {
        "gt_xyxy_orig": gt_xyxy_orig,
        "rt_xyxy_orig": rt_xyxy_orig,
        "abs_err": abs_err,
        "max_abs_err": max_abs_err,
        "iou": iou,
    }




def roundtrip_letterbox_sanity(
    *,
    dataset,
    fpn_meta_npz: str,
    gt_field: str = "ground_truth",
    num_checks: int = 10,
    rng_seed: int = 0,
):
    # uses your meta to define the letterbox size new_h/new_w
    _, meta = load_fpn_memmap(fpn_meta_npz)
    N, C, Hf, Wf = meta["shape"]
    stride = int(meta["stride"])
    new_h = int(Hf * stride)
    new_w = int(Wf * stride)

    sample_ids = meta["sample_ids"]
    view = dataset.select(sample_ids.tolist())
    if view.first().metadata is None:
        view.compute_metadata()

    rng = np.random.default_rng(rng_seed)
    idxs = rng.choice(len(view), size=min(num_checks, len(view)), replace=False)

    worst = None
    for idx in idxs:
        s = view.skip(int(idx)).first()
        dets = s[f"{gt_field}"].detections if s[gt_field] is not None else []
        if not dets:
            continue

        d = dets[0]  # just check first box; you can iterate more if you want
        det_dict = {"bounding_box": d.bounding_box, "label": d.label}

        print("\n--- sample", s.id, "label:", det_dict["label"], "---")
        out = roundtrip_letterbox_one_box(
            dataset=view,
            sample=s,
            det=det_dict,
            new_h=new_h,
            new_w=new_w,
        )

        if worst is None or out["max_abs_err"] > worst["max_abs_err"]:
            worst = out

    print("\nWorst max_abs_err(px):", None if worst is None else worst["max_abs_err"])
    return worst


train_view = fo.load_dataset(TRAIN_DATASET_NAME).load_saved_view(TRAIN_VIEW_NAME)
roundtrip_letterbox_sanity(
    dataset=train_view,
    fpn_meta_npz="./data/openimages_v7/yolo_world_fpn_P4_fp16_train_meta.npz",
    gt_field="ground_truth",
    num_checks=20,
)