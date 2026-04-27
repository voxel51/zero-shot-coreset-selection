import numpy as np

import fiftyone as fo

from detector import load_det_npz, make_class_maps, _xywh_to_xyxy, _pairwise_iou_xyxy, load_fpn_memmap

def fn_breakdown(dataset, pred_npz_path, *, gt_field="ground_truth", iou_thresh=0.25):
    pred = load_det_npz(pred_npz_path)
    classes = pred["classes"]
    class_to_idx, _ = make_class_maps(classes)

    view = dataset.select(pred["sample_ids"].tolist(), ordered=True)
    gt_list = view.values(f"{gt_field}.detections")

    counts = {
        "gt_total": 0,
        "gt_localization_miss": 0,      # max IoU (any class) < thresh
        "gt_class_mismatch": 0,         # max IoU (any class) >= thresh but max IoU (same class) < thresh
        "gt_has_sameclass_overlap": 0,  # max IoU (same class) >= thresh (should usually be matchable)
    }

    for i in range(len(pred["sample_ids"])):
        k = int(pred["num_boxes"][i])
        if k > 0:
            p_xyxy = _xywh_to_xyxy(pred["boxes"][i, :k]).astype(np.float32)
            p_cls = pred["class_idx"][i, :k].astype(np.int32)
            keep = p_cls >= 0
            p_xyxy, p_cls = p_xyxy[keep], p_cls[keep]
        else:
            p_xyxy = np.zeros((0, 4), np.float32)
            p_cls = np.zeros((0,), np.int32)

        gtdets = gt_list[i] or []
        g_boxes, g_cls = [], []
        for d in gtdets:
            if d is None:
                continue
            ci = class_to_idx.get(d["label"], -1)
            if ci < 0:
                continue
            g_boxes.append(d["bounding_box"])
            g_cls.append(ci)

        if not g_boxes:
            continue

        g_xyxy = _xywh_to_xyxy(np.asarray(g_boxes, np.float32))
        g_cls = np.asarray(g_cls, np.int32)

        counts["gt_total"] += len(g_cls)

        if len(p_xyxy) == 0:
            counts["gt_localization_miss"] += len(g_cls)
            continue

        ious = _pairwise_iou_xyxy(p_xyxy, g_xyxy)  # (P,G)

        for gi in range(len(g_cls)):
            max_iou_any = float(ious[:, gi].max())
            same = (p_cls == g_cls[gi])
            max_iou_same = float(ious[same, gi].max()) if np.any(same) else 0.0

            if max_iou_any < iou_thresh:
                counts["gt_localization_miss"] += 1
            elif max_iou_same < iou_thresh:
                counts["gt_class_mismatch"] += 1
            else:
                counts["gt_has_sameclass_overlap"] += 1

    return counts



import numpy as np
from detector import (
    load_fpn_memmap,
    letterbox_params_ultra,
    xywh_norm_to_xyxy_pix,
    map_box_orig_to_letterbox_xyxy_pix,
    xyxy_pix_to_xywh_norm,
)

def count_gt_cell_collisions_for_view(
    view,
    fpn_meta_npz,
    classes,
    *,
    gt_field="ground_truth",
    max_gt_per_image=100,
):
    # Only use meta for grid geometry (Hf,Wf,stride)
    _, meta = load_fpn_memmap(fpn_meta_npz)
    _, _, Hf, Wf = meta["shape"]
    stride = int(meta["stride"])
    new_h, new_w = int(Hf * stride), int(Wf * stride)

    class_to_idx = {c: i for i, c in enumerate(classes)}

    if len(view) == 0:
        return {
            "attempted": 0,
            "kept": 0,
            "dropped": 0,
            "drop_rate": 0.0,
            "per_image": [],
        }

    if view.first().metadata is None:
        view.compute_metadata()

    orig_w = view.values("metadata.width")
    orig_h = view.values("metadata.height")
    gt_list = view.values(f"{gt_field}.detections")

    total_attempted = total_dropped = total_kept = 0
    per_image = []

    for i in range(len(view)):
        used = set()
        attempted = dropped = 0

        W, H = int(orig_w[i]), int(orig_h[i])
        gtdets = gt_list[i] or []

        for d in gtdets[:max_gt_per_image]:
            if d is None:
                continue
            label = d["label"]
            if label not in class_to_idx:
                continue

            attempted += 1

            gt_xyxy_orig = xywh_norm_to_xyxy_pix(d["bounding_box"], W, H)
            gt_xyxy_lb = map_box_orig_to_letterbox_xyxy_pix(
                gt_xyxy_orig, H, W, new_h, new_w
            )
            gt_xywh_lb_norm = xyxy_pix_to_xywh_norm(gt_xyxy_lb, new_w, new_h)

            x_norm, y_norm, w_norm, h_norm = gt_xywh_lb_norm
            cx = (x_norm + 0.5 * w_norm) * new_w
            cy = (y_norm + 0.5 * h_norm) * new_h

            gi = int(np.clip(cx / stride, 0, Wf - 1))
            gj = int(np.clip(cy / stride, 0, Hf - 1))

            key = (gj, gi)
            if key in used:
                dropped += 1
                continue
            used.add(key)

        kept = len(used)
        total_attempted += attempted
        total_dropped += dropped
        total_kept += kept
        per_image.append((attempted, kept, dropped))

    return {
        "attempted": total_attempted,
        "kept": total_kept,
        "dropped": total_dropped,
        "drop_rate": (total_dropped / max(total_attempted, 1)),
        "per_image": per_image,
    }


def fn_breakdown_on_collision_filtered_gt(
    view,
    pred_npz_path,
    fpn_meta_npz,
    *,
    gt_field="ground_truth",
    iou_thresh=0.25,
    max_gt_per_image=100,
):
    pred = load_det_npz(pred_npz_path)
    classes = pred["classes"]
    class_to_idx, _ = make_class_maps(classes)

    # Ensure we iterate in the same sample_id order as predictions
    v = view.select(pred["sample_ids"].tolist(), ordered=True)
    if v.first().metadata is None:
        v.compute_metadata()

    # geometry for cell assignment
    _, meta = load_fpn_memmap(fpn_meta_npz)
    _, _, Hf, Wf = meta["shape"]
    stride = int(meta["stride"])
    new_h, new_w = int(Hf * stride), int(Wf * stride)

    orig_w = v.values("metadata.width")
    orig_h = v.values("metadata.height")
    gt_list = v.values(f"{gt_field}.detections")

    counts = {
        "gt_total": 0,
        "gt_localization_miss": 0,
        "gt_class_mismatch": 0,
        "gt_has_sameclass_overlap": 0,
    }

    for i in range(len(pred["sample_ids"])):
        # preds
        k = int(pred["num_boxes"][i])
        if k > 0:
            p_xyxy = _xywh_to_xyxy(pred["boxes"][i, :k]).astype(np.float32)
            p_cls = pred["class_idx"][i, :k].astype(np.int32)
            keep = p_cls >= 0
            p_xyxy, p_cls = p_xyxy[keep], p_cls[keep]
        else:
            p_xyxy = np.zeros((0, 4), np.float32)
            p_cls = np.zeros((0,), np.int32)

        # collision-filtered GT (same logic as training)
        W, H = int(orig_w[i]), int(orig_h[i])
        used = set()
        g_boxes, g_cls = [], []

        gtdets = gt_list[i] or []
        for d in gtdets[:max_gt_per_image]:
            if d is None:
                continue
            ci = class_to_idx.get(d["label"], -1)
            if ci < 0:
                continue

            gt_xyxy_orig = xywh_norm_to_xyxy_pix(d["bounding_box"], W, H)
            gt_xyxy_lb = map_box_orig_to_letterbox_xyxy_pix(gt_xyxy_orig, H, W, new_h, new_w)
            gt_xywh_lb_norm = xyxy_pix_to_xywh_norm(gt_xyxy_lb, new_w, new_h)

            x_norm, y_norm, w_norm, h_norm = gt_xywh_lb_norm
            cx = (x_norm + 0.5 * w_norm) * new_w
            cy = (y_norm + 0.5 * h_norm) * new_h
            gi = int(np.clip(cx / stride, 0, Wf - 1))
            gj = int(np.clip(cy / stride, 0, Hf - 1))

            key = (gj, gi)
            if key in used:
                continue
            used.add(key)

            g_boxes.append(d["bounding_box"])
            g_cls.append(ci)

        if not g_boxes:
            continue

        g_xyxy = _xywh_to_xyxy(np.asarray(g_boxes, np.float32))
        g_cls = np.asarray(g_cls, np.int32)

        counts["gt_total"] += len(g_cls)

        if len(p_xyxy) == 0:
            counts["gt_localization_miss"] += len(g_cls)
            continue

        ious = _pairwise_iou_xyxy(p_xyxy, g_xyxy)

        for gi in range(len(g_cls)):
            max_iou_any = float(ious[:, gi].max())
            same = (p_cls == g_cls[gi])
            max_iou_same = float(ious[same, gi].max()) if np.any(same) else 0.0

            if max_iou_any < iou_thresh:
                counts["gt_localization_miss"] += 1
            elif max_iou_same < iou_thresh:
                counts["gt_class_mismatch"] += 1
            else:
                counts["gt_has_sameclass_overlap"] += 1

    return counts



TRAIN_DATASET_NAME = "oiv7-detections-train-10k"

TRAIN_VIEW_NAME = "oiv7-detections-train-10k-top20view"


train_ds = fo.load_dataset(TRAIN_DATASET_NAME)

# reload the saved views (same sample subsets)
train_view = train_ds.load_saved_view(TRAIN_VIEW_NAME)

classes = load_det_npz("./data/openimages_v7/yolo_world_dets_train.npz")["classes"]

train_len = len(train_view)
#train_indices = np.random.default_rng(0).choice(train_len, size=train_len, replace=False)
# DEBUG
subset_idx = np.arange(10)
train_fpn_mm, train_fpn_meta = load_fpn_memmap("./data/openimages_v7/yolo_world_fpn_P4_fp16_train_meta.npz")
subset_sample_ids = train_fpn_meta["sample_ids"][subset_idx]

train_view_small = train_ds.select(subset_sample_ids.tolist(), ordered=True)


counts = fn_breakdown(train_view_small, "./data/openimages_v7/tiny_val_preds.npz")

out_small = count_gt_cell_collisions_for_view(
    train_view_small,
    "./data/openimages_v7/yolo_world_fpn_P4_fp16_train_meta.npz",
    classes,
    max_gt_per_image=100,  # match training
)
print(out_small)


print(counts)


counts_kept = fn_breakdown_on_collision_filtered_gt(
    train_view_small,
    "./data/openimages_v7/tiny_val_preds.npz",
    "./data/openimages_v7/yolo_world_fpn_P4_fp16_train_meta.npz",
    iou_thresh=0.25,
    max_gt_per_image=100,
)
print("collision-filtered fn_breakdown:", counts_kept)


from detector import predict_tiny_detector_to_npz

pred_npz_fat = "./data/openimages_v7/tiny_val_preds_fat.npz"



