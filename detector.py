import fiftyone as fo
import numpy as np
import fiftyone.zoo as foz
from fiftyone import ViewField as VF
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import torch.nn as nn   
import contextlib
from ultralytics.models.yolo.detect.predict import DetectionPredictor

# DATASET_NAME = "coco-2017"
DATASET_NAME = "open-images-v7"


TRAIN_DATASET_NAME = "oiv7-detections-train-10k"
VAL_DATASET_NAME = "oiv7-detections-val-500"

TRAIN_VIEW_NAME = "oiv7-detections-train-10k-top20view"
VAL_VIEW_NAME = "oiv7-detections-val-500-top20view"



PROMPT_TEMPLATES: list[str] = [
    "{c}",
    # "a {c}",
    # "the {c}",
    # "{c} object",
    # "{c} in the image",
    # "{c} in the scene",
    # "a photo of {c}",
    # "a photo of a {c}",
    # "a close-up of {c}",
    # "a close-up photo of a {c}",
    # "{c} on a table",
    # "{c} on the ground",
]

def build_prompts(class_name: str, templates: list[str] = PROMPT_TEMPLATES) -> list[str]:
    # Light normalization helps when your labels are like "traffic_light" / "Fire_Hydrant"
    c = class_name.replace("_", " ").strip()
    return [t.format(c=c) for t in templates]


def build_prompt_ensemble(class_names, templates=PROMPT_TEMPLATES):
    prompt_classes = []
    prompt_to_base = {}

    for base in class_names:
        base_norm = base.replace("_", " ").strip()
        for p in build_prompts(base_norm, templates=templates):
            prompt_classes.append(p)
            prompt_to_base[p] = base_norm

    # de-dup while preserving order
    prompt_classes = list(dict.fromkeys(prompt_classes))
    return prompt_classes, prompt_to_base



def _dataset_characteristics():
    coco_train = foz.load_zoo_dataset(DATASET_NAME, split="train", max_samples=5_000)

    print("label fields:", coco_train._get_label_fields())
    det_field = "ground_truth"  # COCO zoo default

    # Make sure width/height exist for pixel conversion
    s = coco_train.first()
    if s.metadata is None:
        coco_train.compute_metadata()
        s = coco_train.first()

    dets = s[det_field]  # fo.Detections
    print("type:", type(dets))
    print("num detections in sample:", len(dets.detections) if dets else 0)

    d = dets.detections[0]
    print("one detection label:", d.label)
    print("one detection bounding_box:", d.bounding_box)  # [x, y, w, h] normalized
    print("confidence/index (may be None):", d.confidence, d.index)

    W, H = s.metadata.width, s.metadata.height
    x, y, w, h = d.bounding_box
    x1, y1, x2, y2 = x * W, y * H, (x + w) * W, (y + h) * H
    print("pixel box (x1,y1,x2,y2):", (x1, y1, x2, y2))


def _get_class_counts(dataset):
    print("label fields:", dataset._get_label_fields())

    det_field = "ground_truth"  # COCO zoo default
    label_path = f"{det_field}.detections.label"

    counts = dataset.count_values(label_path)  # label -> number of boxes
    num_classes = len(counts)
    total_boxes = sum(counts.values())

    print("num classes:", num_classes)
    print("total boxes:", total_boxes)

    for label, n in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:30]:
        print(f"{label:20s} {n:9d}  ({n/total_boxes:.2%})")

    return counts 


def class_distributions_coco():

    print("COCO")
    print()
    train = foz.load_zoo_dataset(DATASET_NAME, split="train", max_samples=5_000)
    val = foz.load_zoo_dataset(DATASET_NAME, split="validation")

    print(f"train: {len(train)} samples")
    _get_class_counts(train)
    print()
    print(f"val: {len(val)} samples")
    _get_class_counts(val)


def class_distributions_oiv7():
    print("Open Images V7")
    print()
    train = foz.load_zoo_dataset(DATASET_NAME, 
                                 split="train", 
                                 label_types=["detections"], 
                                 max_samples=50_000)
    val = foz.load_zoo_dataset(DATASET_NAME, 
                               split="validation", 
                               label_types=["detections"],
                               max_samples=10_000)

    print(f"train: {len(train)} samples")
    _get_class_counts(train)
    print()
    print(f"val: {len(val)} samples")
    _get_class_counts(val)


def make_class_maps(classes: list[str]):
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = list(classes)
    return class_to_idx, idx_to_class



def detections_to_numpy(
    dataset,
    label_field: str,
    class_to_idx: dict[str, int],
    *,
    max_boxes: int = 300,
    sort_by_conf: bool = True,
):
    view = dataset.select_fields([label_field], ordered=True)
    sample_ids = np.array(view.values("id"), dtype=object)
    n = len(view)

    boxes = np.zeros((n, max_boxes, 4), dtype=np.float32)
    class_idx = np.full((n, max_boxes), -1, dtype=np.int32)
    conf = np.full((n, max_boxes), np.nan, dtype=np.float32)
    num_boxes = np.zeros((n,), dtype=np.int32)

    for i, sample in enumerate(view):
        dets = sample[label_field]
        if dets is None or not dets.detections:
            continue

        detections = list(dets.detections)

        if sort_by_conf:
            detections.sort(
                key=lambda d: (-1.0 if d.confidence is None else float(d.confidence)),
                reverse=True,
            )

        k = min(len(detections), max_boxes)
        num_boxes[i] = k

        for j in range(k):
            d = detections[j]
            boxes[i, j] = np.asarray(d.bounding_box, dtype=np.float32)
            class_idx[i, j] = class_to_idx.get(d.label, -1)
            conf[i, j] = np.nan if d.confidence is None else np.float32(d.confidence)

    return boxes, class_idx, conf, num_boxes, sample_ids


def save_det_npz(path, *, boxes, class_idx, conf, num_boxes, sample_ids, classes):
    np.savez_compressed(
        path,
        boxes=boxes,
        class_idx=class_idx,
        conf=conf,
        num_boxes=num_boxes,
        sample_ids=sample_ids,
        classes=np.array(classes, dtype=object),
    )

def load_det_npz(path):
    d = np.load(path, allow_pickle=True)
    classes = d["classes"].tolist()
    class_to_idx, idx_to_class = make_class_maps(classes)
    return {
        "boxes": d["boxes"],
        "class_idx": d["class_idx"],
        "conf": d["conf"],
        "num_boxes": d["num_boxes"],
        "sample_ids": d["sample_ids"],
        "classes": classes,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
    }


def decode_class_idx(class_idx: np.ndarray, idx_to_class: list[str]):
    idx_to_class_arr = np.array(idx_to_class, dtype=object)
    out = np.empty(class_idx.shape, dtype=object)
    out[:] = None
    mask = class_idx >= 0
    out[mask] = idx_to_class_arr[class_idx[mask]]
    return out




class NoLockPredictor(DetectionPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = contextlib.nullcontext()  # has __enter__/__exit__
        # alternatively: self._lock = threading.RLock()


def extract_fpn_level_memmap(
    fiftyone_yolo_model,   # the object returned by foz.load_zoo_model(...)
    filepaths: list[str],
    sample_ids: np.ndarray,
    *,
    out_memmap_path: str,
    out_meta_npz: str,
    level: int = 1,          # 0=P3, 1=P4, 2=P5; P5 the smallest
    batch_size: int = 16,
    dtype=np.float16,
):

    ultra = fiftyone_yolo_model._model  # ultralytics.YOLO instance

    # torch DetectionModel is ultra.predictor.model; its module list is .model
    #detect = ultra.predictor.model.model[-1]  # Detect()
    detect = ultra.model.model[-1]
    stride = int(detect.stride[level].item()) if hasattr(detect, "stride") else None

    captured = {}

    def _pre_hook(module, inputs):
        # inputs[0] is the list of FPN tensors [P3, P4, P5] with shape (B,C,H,W)
        captured["fpn"] = inputs[0]

    hook = detect.register_forward_pre_hook(_pre_hook)

    N = len(filepaths)
    mm = None
    c_hw = None

    for start in range(0, N, batch_size):
        batch_paths = filepaths[start : start + batch_size]

        # runs preprocessing (letterbox to imgsz) + inference; hook fires
        # _ = ultra.predict(batch_paths, verbose=False)
        ultra.predictor = None
        _ = ultra.predict(batch_paths, verbose=False, predictor=NoLockPredictor)

        # DEBUG
        print("predictor class now:", type(ultra.predictor))
        print("_lock:", ultra.predictor._lock)

        fpn_list = captured.pop("fpn")              # list of tensors
        f = fpn_list[level].detach().cpu().numpy() # (B,C,H,W)
        f = f.astype(dtype, copy=False)

        if mm is None:
            _, C, H, W = f.shape
            c_hw = (C, H, W)
            mm = np.memmap(out_memmap_path, mode="w+", dtype=dtype, shape=(N, C, H, W))

        mm[start : start + f.shape[0]] = f

    hook.remove()
    mm.flush()

    np.savez_compressed(
        out_meta_npz,
        memmap_path=np.array(out_memmap_path, dtype=object),
        sample_ids=sample_ids,
        level=np.int32(level),
        stride=np.int32(-1 if stride is None else stride),
        shape=np.array((N, *c_hw), dtype=np.int32),
        dtype=np.array(str(np.dtype(dtype)), dtype=object),
    )

    return out_memmap_path, out_meta_npz


def yolo_world_embeddings(
    *,
    label_field: str = "yolo_world_detections",
    max_boxes: int = 30,
    out_npz: str = "./data/openimages_v7/yolo_world_dets_train.npz",
):
    train_dataset = foz.load_zoo_dataset(
        DATASET_NAME,
        split="train",
        max_samples=10_000,
        label_types=["detections"],
        dataset_name=TRAIN_DATASET_NAME,
        persistent=True,
    )

    val_dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="validation",
        label_types=["detections"],
        max_samples=500,
        dataset_name=VAL_DATASET_NAME,
        persistent=True,
    )

    # 1) Canonical class list (boxable, stable ordering)
    classes_train = list(train_dataset.info["classes_map"].values())
    counts_train = _get_class_counts(train_dataset)

    # restrict to 20 most common classes
    classes = [k for k, v in sorted(counts_train.items(), key=lambda kv: kv[1], reverse=True)[:20]]    
    # Filter samples that contain one of the top classes
    train_view = train_dataset.match(
        VF("ground_truth.detections").filter(VF("label").is_in(classes)).length() > 0
    )
    train_dataset.save_view(TRAIN_VIEW_NAME, train_view, overwrite=True)

    class_to_idx, idx_to_class = make_class_maps(classes)

    val_view = val_dataset.match(
        VF("ground_truth.detections").filter(VF("label").is_in(classes)).length() > 0
    )
    val_dataset.save_view(VAL_VIEW_NAME, val_view, overwrite=True)

    #import ipdb; ipdb.set_trace()


    # 2) Run YOLO-World with prompt variants, then collapse back to base labels
    prompt_classes, prompt_to_base = build_prompt_ensemble(classes)
    model = foz.load_zoo_model("yolov8s-world-torch", classes=prompt_classes)

    train_view.apply_model(model, label_field=label_field)
    train_view = train_view.map_labels(label_field, prompt_to_base)
    train_view.save(fields=label_field)  # now stored labels are base class strings
    filepaths = train_view.values("filepath")
    sample_ids = np.array(train_view.values("id"), dtype=object)

    extract_fpn_level_memmap(
        model,
        filepaths,
        sample_ids,
        out_memmap_path="./data/openimages_v7/yolo_world_fpn_P4_fp16_train.dat",
        out_meta_npz="./data/openimages_v7/yolo_world_fpn_P4_fp16_train_meta.npz",
        level=1,          # P4 only (usually a good “middle” scale)
        batch_size=16,
    )

    # (optional but helps long-term consistency)
    train_view.classes[label_field] = classes
    train_view.save()

    # 3) Extract fixed-shape numpy + save (includes `classes`)
    boxes, class_idx, conf, num_boxes, sample_ids = detections_to_numpy(
        train_view, label_field, class_to_idx, max_boxes=max_boxes
    )
    save_det_npz(
        out_npz,
        boxes=boxes,
        class_idx=class_idx,
        conf=conf,
        num_boxes=num_boxes,
        sample_ids=sample_ids,
        classes=classes,
    )

    filepaths = val_view.values("filepath")
    sample_ids = np.array(val_view.values("id"), dtype=object)

    extract_fpn_level_memmap(
        model,
        filepaths,
        sample_ids,
        out_memmap_path="./data/openimages_v7/yolo_world_fpn_P4_fp16_val.dat",
        out_meta_npz="./data/openimages_v7/yolo_world_fpn_P4_fp16_val_meta.npz",
        level=1,
        batch_size=16,
        dtype=np.float16,
    )

    return out_npz



def _xywh_to_xyxy(boxes_xywh: np.ndarray) -> np.ndarray:
    x, y, w, h = np.split(boxes_xywh, 4, axis=-1)
    return np.concatenate([x, y, x + w, y + h], axis=-1)

def _pairwise_iou_xyxy(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = np.maximum(0.0, (ax2 - ax1)) * np.maximum(0.0, (ay2 - ay1))
    area_b = np.maximum(0.0, (bx2 - bx1)) * np.maximum(0.0, (by2 - by1))
    union = area_a + area_b - inter
    return inter / (union + eps)

def eval_det_f1_top1(
    dataset,
    pred_npz_path: str,
    *,
    gt_field: str = "ground_truth",
    iou_thresh: float = 0.25,
    conf_thresh: float | None = None,  # NEW: set e.g. 0.3 or 0.5
):
    pred = load_det_npz(pred_npz_path)
    classes = pred["classes"]
    class_to_idx, _ = make_class_maps(classes)

    pred_boxes = pred["boxes"]      # (N,M,4) xywh
    pred_cls = pred["class_idx"]    # (N,M)
    pred_conf = pred["conf"]        # (N,M)  NEW
    pred_n = pred["num_boxes"]      # (N,)
    sample_ids = pred["sample_ids"] # (N,)

    # Load GT in same sample-id order
    view = dataset.select(sample_ids.tolist(), ordered=True)
    gt_dets_list = view.values(f"{gt_field}.detections")

    TP = FP = FN = 0

    for i in range(len(sample_ids)):
        k = int(pred_n[i])
        if k > 0:
            p_xyxy = _xywh_to_xyxy(pred_boxes[i, :k])
            p_cls = pred_cls[i, :k].astype(np.int32)
            keep_p = p_cls >= 0
            p_xyxy = p_xyxy[keep_p]
            p_cls  = p_cls[keep_p]
            p_conf = pred_conf[i, :k].astype(np.float32)
            p_conf = p_conf[keep_p]

            # NEW: confidence filtering (also drops NaNs)
            if conf_thresh is not None:
                keep = np.isfinite(p_conf) & (p_conf >= float(conf_thresh))
                if keep.any():
                    p_xyxy = p_xyxy[keep]
                    p_cls = p_cls[keep]
                else:
                    p_xyxy = np.zeros((0, 4), dtype=np.float32)
                    p_cls = np.zeros((0,), dtype=np.int32)
        else:
            p_xyxy = np.zeros((0, 4), dtype=np.float32)
            p_cls = np.zeros((0,), dtype=np.int32)

        gtdets = gt_dets_list[i] or []
        g_boxes = []
        g_cls = []
        for d in gtdets:
            if d is None:
                continue
            g_boxes.append(d["bounding_box"])
            g_cls.append(class_to_idx.get(d["label"], -1))

        if len(g_boxes) > 0:
            g_xyxy = _xywh_to_xyxy(np.asarray(g_boxes, dtype=np.float32))
            g_cls = np.asarray(g_cls, dtype=np.int32)
            keep_g = g_cls >= 0
            g_xyxy = g_xyxy[keep_g]
            g_cls  = g_cls[keep_g]
        else:
            g_xyxy = np.zeros((0, 4), dtype=np.float32)
            g_cls = np.zeros((0,), dtype=np.int32)

        if p_xyxy.shape[0] == 0 and g_xyxy.shape[0] == 0:
            continue
        if p_xyxy.shape[0] == 0:
            FN += g_xyxy.shape[0]
            continue
        if g_xyxy.shape[0] == 0:
            FP += p_xyxy.shape[0]
            continue

        ious = _pairwise_iou_xyxy(p_xyxy, g_xyxy)  # (P,G)
        cand = np.argwhere(ious >= iou_thresh)
        cand = sorted(cand, key=lambda ij: float(ious[ij[0], ij[1]]), reverse=True)

        matched_p = set()
        matched_g = set()

        for pi, gi in cand:
            if pi in matched_p or gi in matched_g:
                continue
            if p_cls[pi] != g_cls[gi]:
                continue
            matched_p.add(int(pi))
            matched_g.add(int(gi))

        TP += len(matched_p)
        FP += (p_xyxy.shape[0] - len(matched_p))
        FN += (g_xyxy.shape[0] - len(matched_g))

    precision = TP / (TP + FP + 1e-12)
    recall = TP / (TP + FN + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return {"TP": TP, "FP": FP, "FN": FN, "precision": precision, "recall": recall, "f1": f1}

def add_xy_coords(x: torch.Tensor) -> torch.Tensor:
    # x: (B,C,H,W) -> (B,C+2,H,W), coords in [0,1]
    B, _, H, W = x.shape
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, H, device=x.device, dtype=x.dtype),
        torch.linspace(0, 1, W, device=x.device, dtype=x.dtype),
        indexing="ij",
    )
    coords = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(B, 2, H, W)
    return torch.cat([x, coords], dim=1)


def load_fpn_memmap(meta_npz_path: str):
    meta = np.load(meta_npz_path, allow_pickle=True)
    memmap_path = str(meta["memmap_path"].item())
    shape = tuple(meta["shape"].tolist())  # (N,C,H,W)
    dtype = np.dtype(str(meta["dtype"].item()))
    stride = int(meta["stride"].item())
    level = int(meta["level"].item())
    sample_ids = meta["sample_ids"]  # object array

    mm = np.memmap(memmap_path, mode="r", dtype=dtype, shape=shape)
    return mm, {"shape": shape, "dtype": dtype, "stride": stride, "level": level, "sample_ids": sample_ids}

# def letterbox_params_ultra(orig_h: int, orig_w: int, new_h: int, new_w: int):
#     # Matches Ultralytics LetterBox when auto=False, scale_fill=False, center=True, scaleup=True
#     r = min(new_h / orig_h, new_w / orig_w)
#     new_unpad_w = int(round(orig_w * r))
#     new_unpad_h = int(round(orig_h * r))
#     dw = (new_w - new_unpad_w) / 2
#     dh = (new_h - new_unpad_h) / 2
#     left = int(round(dw - 0.1))
#     top  = int(round(dh - 0.1))
#     return r, left, top, new_unpad_w, new_unpad_h

def letterbox_params_ultra(orig_h: int, orig_w: int, new_h: int, new_w: int, *, stride: int = 32, auto: bool = False):
    # Mimics Ultralytics LetterBox() behavior for inference
    r = min(new_h / orig_h, new_w / orig_w)  # scale
    new_unpad_w = int(round(orig_w * r))
    new_unpad_h = int(round(orig_h * r))

    dw = new_w - new_unpad_w
    dh = new_h - new_unpad_h

    if auto:
        dw = dw % stride
        dh = dh % stride

    dw /= 2
    dh /= 2

    left = int(round(dw - 0.1))
    top  = int(round(dh - 0.1))
    return r, left, top, new_unpad_w, new_unpad_h

def xywh_norm_to_xyxy_pix(box_xywh_norm, W, H):
    x, y, w, h = box_xywh_norm
    x1 = x * W
    y1 = y * H
    x2 = (x + w) * W
    y2 = (y + h) * H
    return np.array([x1, y1, x2, y2], dtype=np.float32)

def xyxy_pix_to_xywh_norm(box_xyxy_pix, W, H):
    x1, y1, x2, y2 = box_xyxy_pix
    x1 = np.clip(x1, 0, W)
    x2 = np.clip(x2, 0, W)
    y1 = np.clip(y1, 0, H)
    y2 = np.clip(y2, 0, H)
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    x = x1
    y = y1
    return np.array([x / W, y / H, w / W, h / H], dtype=np.float32)

def map_box_orig_to_letterbox_xyxy_pix(box_xyxy_orig_pix, orig_h, orig_w, new_h, new_w):
    r, left, top, _, _ = letterbox_params_ultra(orig_h, orig_w, new_h, new_w)
    x1, y1, x2, y2 = box_xyxy_orig_pix
    return np.array([x1 * r + left, y1 * r + top, x2 * r + left, y2 * r + top], dtype=np.float32)

def map_box_letterbox_to_orig_xyxy_pix(box_xyxy_lb_pix, orig_h, orig_w, new_h, new_w):
    r, left, top, _, _ = letterbox_params_ultra(orig_h, orig_w, new_h, new_w)
    x1, y1, x2, y2 = box_xyxy_lb_pix
    x1 = (x1 - left) / r
    x2 = (x2 - left) / r
    y1 = (y1 - top) / r
    y2 = (y2 - top) / r
    return np.array([x1, y1, x2, y2], dtype=np.float32)



class FpnGtDataset(Dataset):
    def __init__(
        self,
        *,
        fpn_memmap,             # np.memmap (N,C,H,W)
        fpn_meta: dict,         # from load_fpn_memmap
        dataset,                # FiftyOne dataset with GT
        classes: list[str],     # same classes list used in your det npz
        gt_field: str = "ground_truth",
        max_gt_per_image: int = 100,
    ):
        self.fpn = fpn_memmap
        self.N, self.C, self.Hf, self.Wf = fpn_meta["shape"]
        self.stride = int(fpn_meta["stride"])
        self.sample_ids = fpn_meta["sample_ids"]
        self.gt_field = gt_field
        self.max_gt = max_gt_per_image

        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        # Ensure metadata exists on samples (width/height)
        view = dataset.select(self.sample_ids.tolist(), ordered=True)
        if view.first().metadata is None:
            view.compute_metadata()

        self.view = view
        self.orig_wh = list(zip(view.values("metadata.width"), view.values("metadata.height")))
        self.gt = view.values(f"{gt_field}.detections")  # list-of-lists of detections

        # derive letterbox input size from fpn + stride
        self.new_h = int(self.Hf * self.stride)
        self.new_w = int(self.Wf * self.stride)

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        x = np.asarray(self.fpn[i], dtype=np.float32)  # (C,Hf,Wf)
        x = torch.from_numpy(x)

        W, H = self.orig_wh[i]
        gtdets = self.gt[i] or []

        obj = np.zeros((self.Hf, self.Wf), dtype=np.float32)
        cls = np.zeros((self.Hf, self.Wf), dtype=np.int64)
        box = np.zeros((4, self.Hf, self.Wf), dtype=np.float32)

        used = set()  # prevent multiple GT assigned to same cell; keep the first (or you can keep largest)

        attempted = 0
        dropped = 0

        for d in gtdets[: self.max_gt]:
            if d is None:
                continue
            label = d["label"]
            if label not in self.class_to_idx:
                continue

            # original normalized -> original pixels xyxy
            gt_xyxy_orig = xywh_norm_to_xyxy_pix(d["bounding_box"], W, H)

            # map to letterboxed pixel coords
            gt_xyxy_lb = map_box_orig_to_letterbox_xyxy_pix(gt_xyxy_orig, H, W, self.new_h, self.new_w)

            # convert to letterboxed normalized xywh
            gt_xywh_lb_norm = xyxy_pix_to_xywh_norm(gt_xyxy_lb, self.new_w, self.new_h)
            x_norm, y_norm, w_norm, h_norm = gt_xywh_lb_norm
            cx = (x_norm + 0.5 * w_norm) * self.new_w
            cy = (y_norm + 0.5 * h_norm) * self.new_h

            gi = int(np.clip(cx / self.stride, 0, self.Wf - 1))
            gj = int(np.clip(cy / self.stride, 0, self.Hf - 1))


            attempted += 1
            key = (gj, gi)
            if key in used:
                dropped += 1
                continue
            used.add(key)

            obj[gj, gi] = 1.0
            cls[gj, gi] = int(self.class_to_idx[label])
            box[:, gj, gi] = gt_xywh_lb_norm  # normalized to letterboxed input


        # print(f"attempted: {attempted}, dropped: {dropped}")


        return x, torch.from_numpy(obj), torch.from_numpy(cls), torch.from_numpy(box)
    


class TinyFpnDetector(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.obj_head = nn.Conv2d(hidden, 1, 1)            # (B,1,H,W)
        self.cls_head = nn.Conv2d(hidden, num_classes, 1)  # (B,C,H,W)
        self.box_head = nn.Conv2d(hidden, 4, 1)            # (B,4,H,W) predicts normalized xywh in [0,1]

    def forward(self, x):
        f = self.trunk(x)
        obj_logits = self.obj_head(f).squeeze(1)           # (B,H,W)
        cls_logits = self.cls_head(f)                      # (B,C,H,W)
        box_raw = self.box_head(f)                         # (B,4,H,W)
        box_xywh = torch.sigmoid(box_raw)                  # keep it simple/consistent
        return obj_logits, cls_logits, box_xywh



def train_tiny_detector(
    *,
    fpn_meta_npz: str,
    dataset,                       # FiftyOne dataset containing GT
    classes: list[str],
    device: str = "cuda",
    gt_field: str = "ground_truth",
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    obj_pos_weight: float = 10.0,  # because most cells are negatives
):
    fpn_mm, fpn_meta = load_fpn_memmap(fpn_meta_npz)

    ds = FpnGtDataset(
        fpn_memmap=fpn_mm,
        fpn_meta=fpn_meta,
        dataset=dataset,
        classes=classes,
        gt_field=gt_field,
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    model = TinyFpnDetector(in_ch=ds.C+2, num_classes=len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(obj_pos_weight, device=device))

    model.train()
    for ep in range(epochs):
        tot = 0.0
        for x, obj_t, cls_t, box_t in dl:
            x = x.to(device)
            x = add_xy_coords(x)  # (B,C+2,H,W)
            obj_t = obj_t.to(device)         # (B,H,W)
            cls_t = cls_t.to(device)         # (B,H,W)
            box_t = box_t.to(device)         # (B,4,H,W)

            obj_logits, cls_logits, box_xywh = model(x)

            # objectness loss over all cells
            loss_obj = bce(obj_logits, obj_t)

            # only compute cls/box loss on positive cells
            pos = obj_t > 0.5
            if pos.any():
                # cls_logits: (B,C,H,W) -> (num_pos,C)
                cls_pos = cls_logits.permute(0, 2, 3, 1)[pos]  # (P,C)
                cls_gt = cls_t[pos]                            # (P,)
                loss_cls = F.cross_entropy(cls_pos, cls_gt)

                box_pos = box_xywh.permute(0, 2, 3, 1)[pos]    # (P,4)
                box_gt = box_t.permute(0, 2, 3, 1)[pos]        # (P,4)
                loss_box = F.l1_loss(box_pos, box_gt)
            else:
                loss_cls = torch.tensor(0.0, device=device)
                loss_box = torch.tensor(0.0, device=device)

            loss = loss_obj + loss_cls + loss_box

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            tot += float(loss.item())

        print(f"epoch {ep+1}/{epochs} loss={tot/len(dl):.4f}")

    return model, fpn_meta


def nms_xyxy(boxes, scores, iou_thresh=0.5):
    # boxes: (K,4) xyxy in pixels, scores: (K,)
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2-xx1)
        h = np.maximum(0, yy2-yy1)
        inter = w*h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=np.int64)


@torch.no_grad()
def predict_tiny_detector_to_npz(
    *,
    model,
    fpn_meta_npz: str,
    dataset,  # FiftyOne dataset/view for metadata (orig W/H)
    classes: list[str],
    out_npz: str,
    device: str = "cuda",
    max_boxes: int = 200,
    score_thresh: float = 0.05,
    nms_iou: float = 0.9,
    indices: np.ndarray | list[int] | None = None,   # NEW
):
    fpn_mm, fpn_meta = load_fpn_memmap(fpn_meta_npz)
    N_full, C, Hf, Wf = fpn_meta["shape"]
    stride = int(fpn_meta["stride"])
    if stride <= 0:
        raise ValueError(f"Invalid stride in fpn meta: {stride}")
    sample_ids_full = fpn_meta["sample_ids"]

    if indices is None:
        indices = np.arange(N_full, dtype=np.int64)
    else:
        indices = np.asarray(indices, dtype=np.int64)

    # subset sample IDs in the same order as indices
    sample_ids = sample_ids_full[indices]

    # pull widths/heights for exactly these sample_ids (and in this order)
    view = dataset.select(sample_ids.tolist(), ordered=True)
    if len(view) != len(sample_ids):
        raise ValueError(
            f"Requested {len(sample_ids)} sample_ids but view has {len(view)}. "
            "This usually means your FiftyOne view/dataset doesn't contain all sample_ids "
            "from the FPN meta subset."
        )

    if view.first().metadata is None:
        view.compute_metadata()
    orig_w = view.values("metadata.width")
    orig_h = view.values("metadata.height")

    new_h = int(Hf * stride)
    new_w = int(Wf * stride)

    N = len(indices)
    boxes_out = np.zeros((N, max_boxes, 4), dtype=np.float32)
    cls_out = np.full((N, max_boxes), -1, dtype=np.int32)
    conf_out = np.full((N, max_boxes), np.nan, dtype=np.float32)
    num_out = np.zeros((N,), dtype=np.int32)

    model.eval().to(device)

    pre_nms_topk = max(max_boxes * 50, 200)

    for out_i, mm_i in enumerate(indices):
        x = torch.from_numpy(np.asarray(fpn_mm[mm_i], dtype=np.float32)).unsqueeze(0).to(device)
        x = add_xy_coords(x)

        obj_logits, cls_logits, box_xywh = model(x)

        obj_prob = torch.sigmoid(obj_logits[0])                 # (Hf,Wf)
        cls_prob = torch.softmax(cls_logits[0], dim=0)          # (C,Hf,Wf)
        max_cls_prob, cls_idx_t = cls_prob.max(dim=0)           # (Hf,Wf), (Hf,Wf)

        score_t = obj_prob * max_cls_prob
        score_map = score_t.detach().cpu().numpy().astype(np.float32)
        cls_idx_map = cls_idx_t.detach().cpu().numpy().astype(np.int32)
        box = box_xywh[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)

        flat = score_map.reshape(-1)
        cand = np.flatnonzero(flat >= float(score_thresh))
        if cand.size == 0:
            continue

        if cand.size > pre_nms_topk:
            part = np.argpartition(flat[cand], -pre_nms_topk)[-pre_nms_topk:]
            cand = cand[part]

        cand = cand[np.argsort(flat[cand])[::-1]]

        ys = (cand // Wf).astype(np.int32)
        xs = (cand % Wf).astype(np.int32)
        scores = flat[cand].astype(np.float32)

        xywh = box[ys, xs]
        x1y1x2y2_lb = _xywh_to_xyxy(xywh) * np.array([new_w, new_h, new_w, new_h], dtype=np.float32)

        cand_cls = cls_idx_map[ys, xs].astype(np.int32)

        keep_all = []
        for c in np.unique(cand_cls):
            m = (cand_cls == c)
            if not np.any(m):
                continue
            keep_c = nms_xyxy(x1y1x2y2_lb[m], scores[m], iou_thresh=nms_iou)
            keep_all.append(np.flatnonzero(m)[keep_c])

        if len(keep_all) == 0:
            continue

        keep = np.concatenate(keep_all, axis=0)
        keep = keep[np.argsort(scores[keep])[::-1]][:max_boxes]

        k = int(keep.size)
        num_out[out_i] = k
        if k == 0:
            continue

        ys = ys[keep]
        xs = xs[keep]
        scores = scores[keep]
        xywh = xywh[keep]
        pred_cls = cand_cls[keep]

        W0, H0 = int(orig_w[out_i]), int(orig_h[out_i])

        for j in range(k):
            lb_xyxy_pix = _xywh_to_xyxy(xywh[j : j + 1])[0] * np.array(
                [new_w, new_h, new_w, new_h], dtype=np.float32
            )
            orig_xyxy_pix = map_box_letterbox_to_orig_xyxy_pix(lb_xyxy_pix, H0, W0, new_h, new_w)
            orig_xywh_norm = xyxy_pix_to_xywh_norm(orig_xyxy_pix, W0, H0)

            boxes_out[out_i, j] = orig_xywh_norm
            cls_out[out_i, j] = int(pred_cls[j])
            conf_out[out_i, j] = float(scores[j])

    save_det_npz(
        out_npz,
        boxes=boxes_out,
        class_idx=cls_out,
        conf=conf_out,
        num_boxes=num_out,
        sample_ids=np.array(sample_ids, dtype=object),  # IMPORTANT: subset IDs
        classes=classes,
    )
    return out_npz


def _sanity_check():
    pred = load_det_npz("./data/openimages_v7/yolo_world_dets_train.npz")
    M = pred["class_idx"].shape[1]
    mask = (np.arange(M)[None, :] < pred["num_boxes"][:, None])  # valid slots only
    valid_cls = pred["class_idx"][mask]
    print("fraction valid predicted boxes with class_idx == -1:", (valid_cls < 0).mean())

    print("avg boxes/image:", pred["num_boxes"].mean(), "median:", np.median(pred["num_boxes"]))


    valid = (np.arange(M)[None, :] < pred["num_boxes"][:, None])

    for thr in [0.05, 0.1, 0.2, 0.3, 0.5]:
        keep = valid & np.isfinite(pred["conf"]) & (pred["conf"] >= thr)
        avg_kept = keep.sum() / pred["num_boxes"].sum()
        print("thr", thr, "fraction of valid boxes kept", avg_kept)


    dataset = foz.load_zoo_dataset(
        DATASET_NAME,
        split="train",
        max_samples=5_000,
        label_types=["detections"],
        dataset_name="oiv7-detections-train-5k",
        persistent=True,
    )  

    label_field = "yolo_world_detections"  # or whatever you used

    classes = list(dataset.info["classes_map"].values())
    class_set = set(classes)

    pred_counts = dataset.count_values(f"{label_field}.detections.label")
    not_in_classes = [(k, v) for k, v in pred_counts.items() if k not in class_set]
    not_in_classes.sort(key=lambda kv: kv[1], reverse=True)

    print("num predicted label strings not in classes:", len(not_in_classes))
    print("top 30 unmapped predicted labels:", not_in_classes[:30])

    counts = dataset.count_values("ground_truth.detections.label")
    missing = [k for k in counts.keys() if k not in pred["class_to_idx"]]
    print("GT labels missing from class map:", len(missing))
    print("example missing:", missing[:20])


def train_on_fpn_indices_and_eval(
    *,
    # --- training inputs ---
    train_dataset,                 # FiftyOne dataset with GT for training split
    train_fpn_meta_npz: str,       # meta for train memmap
    train_indices: list[int] | np.ndarray,

    # --- evaluation inputs ---
    val_dataset,                   # FiftyOne dataset with GT for val split
    val_fpn_meta_npz: str,         # meta for val memmap (must correspond to val_dataset samples)
    classes: list[str],            # stable class list used for cls head + eval mapping

    # --- hyperparams ---
    device: str = "cuda",
    gt_field: str = "ground_truth",
    epochs: int = 50,
    batch_size: int = 32,
    # batch_size: int = 5,
    lr: float = 1e-3,
    obj_pos_weight: float = 10.0,

    # --- prediction/eval params ---
    pred_score_thresh: float = 0.3,
    pred_nms_iou: float = 0.25,
    #pred_nms_iou: float = 1.0,
    eval_iou_thresh: float = 0.25,
    eval_conf_thresh: float | None = None,
    out_val_pred_npz: str = "./data/openimages_v7/tiny_val_preds.npz",
):
    # 1) Load train memmap + build train dataset
    train_fpn_mm, train_fpn_meta = load_fpn_memmap(train_fpn_meta_npz)
    train_ds_full = FpnGtDataset(
        fpn_memmap=train_fpn_mm,
        fpn_meta=train_fpn_meta,
        dataset=train_dataset,
        classes=classes,
        gt_field=gt_field,
    )

    # 2) Subset by indices
    idx = np.asarray(train_indices, dtype=np.int64)
    train_ds = Subset(train_ds_full, idx.tolist())
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    # 3) Train model (same logic as `train_tiny_detector`, but using our subset loader)
    model = TinyFpnDetector(in_ch=train_ds_full.C+2, num_classes=len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(obj_pos_weight, device=device))

    model.train()
    for ep in range(epochs):
        tot = 0.0
        for x, obj_t, cls_t, box_t in train_dl:
            x = x.to(device)
            x = add_xy_coords(x)  # (B,C+2,H,W)
            obj_t = obj_t.to(device)
            cls_t = cls_t.to(device)
            box_t = box_t.to(device)

            obj_logits, cls_logits, box_xywh = model(x)

            loss_obj = bce(obj_logits, obj_t)

            pos = obj_t > 0.5
            if pos.any():
                cls_pos = cls_logits.permute(0, 2, 3, 1)[pos]
                cls_gt = cls_t[pos]
                loss_cls = torch.nn.functional.cross_entropy(cls_pos, cls_gt)

                box_pos = box_xywh.permute(0, 2, 3, 1)[pos]
                box_gt = box_t.permute(0, 2, 3, 1)[pos]
                loss_box = torch.nn.functional.l1_loss(box_pos, box_gt)
            else:
                loss_cls = torch.tensor(0.0, device=device)
                loss_box = torch.tensor(0.0, device=device)

            loss = loss_obj + loss_cls + loss_box
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.item())

        print(f"epoch {ep+1}/{epochs} loss={tot/len(train_dl):.4f}")


    for pred_score_thres in [0.00, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3]:
        print(f"Evaluating with pred_score_thresh={pred_score_thres}...")
        # 4) Predict on val FPN memmap, write val preds npz
        val_pred_npz = predict_tiny_detector_to_npz(
            model=model,
            fpn_meta_npz=val_fpn_meta_npz,
            dataset=val_dataset,
            classes=classes,
            out_npz=out_val_pred_npz,
            device=device,
            score_thresh=pred_score_thres,
            nms_iou=pred_nms_iou,
            indices=train_indices,
        )

        # 5) Evaluate against val GT
        metrics = eval_det_f1_top1(
            val_dataset,
            val_pred_npz,
            gt_field=gt_field,
            iou_thresh=eval_iou_thresh,
            conf_thresh=eval_conf_thresh,
        )
        print(metrics)
        print()

    return model, metrics



def main():
    # _dataset_characteristics()
    #class_distributions_coco()
    #class_distributions_oiv7()
    #yolo_world_embeddings()

    # reload the exact persistent datasets created by yolo_world_embeddings()
    # train_ds = fo.load_dataset(TRAIN_DATASET_NAME)
    # val_ds   = fo.load_dataset(VAL_DATASET_NAME)

    # # reload the saved views (same sample subsets)
    # train_view = train_ds.load_saved_view(TRAIN_VIEW_NAME)
    # val_view   = val_ds.load_saved_view(VAL_VIEW_NAME) 

    # for conf_thres in [0.05, 0.1, 0.2, 0.3, 0.5]:
    #     print(f"Evaluating with conf_thres={conf_thres}...")
    #     metrics = eval_det_f1_top1(train_view, 
    #                                "./data/openimages_v7/yolo_world_dets_train.npz", 
    #                                iou_thresh=0.25,
    #                                conf_thresh=conf_thres)
    #     print(metrics)

    train_and_eval()

    # pred_npz = predict_tiny_detector_to_npz(
    #     model=model,
    #     fpn_meta_npz="./data/openimages_v7/yolo_world_fpn_P4_fp16_meta.npz",
    #     dataset=foz.load_zoo_dataset("open-images-v7", split="train", label_types=["detections"], max_samples=5_000),
    #     classes=load_det_npz("./data/openimages_v7/yolo_world_dets_train.npz")["classes"],
    #     out_npz="./data/openimages_v7/tiny_head_preds.npz",
    # )

    #     metrics = eval_det_f1_top1(
    #         foz.load_zoo_dataset("open-images-v7", split="train", label_types=["detections"], max_samples=5_000),
    #         pred_npz,
    #         iou_thresh=0.25,
    #     )
    #     print(metrics)

    # _sanity_check()



def train_and_eval():
    # reload the exact persistent datasets created by yolo_world_embeddings()
    train_ds = fo.load_dataset(TRAIN_DATASET_NAME)
    val_ds   = fo.load_dataset(VAL_DATASET_NAME)

    # reload the saved views (same sample subsets)
    train_view = train_ds.load_saved_view(TRAIN_VIEW_NAME)
    val_view   = val_ds.load_saved_view(VAL_VIEW_NAME) 

    classes = load_det_npz("./data/openimages_v7/yolo_world_dets_train.npz")["classes"]

    train_len = len(train_view)
    train_indices = np.random.default_rng(0).choice(train_len, size=train_len, replace=False)
    # DEBUG
    subset_idx = np.arange(10)
    train_fpn_mm, train_fpn_meta = load_fpn_memmap("./data/openimages_v7/yolo_world_fpn_P4_fp16_train_meta.npz")
    subset_sample_ids = train_fpn_meta["sample_ids"][subset_idx]

    train_view_small = train_ds.select(subset_sample_ids.tolist(), ordered=True)
    # DEBUG END

    model, metrics = train_on_fpn_indices_and_eval(
        train_dataset=train_view,
        #train_dataset=train_view_small,
        train_fpn_meta_npz="./data/openimages_v7/yolo_world_fpn_P4_fp16_train_meta.npz",
        train_indices=train_indices,
        #train_indices=subset_idx,
        #val_dataset=val_view,
        val_dataset=train_view,
        #val_dataset=train_view_small,
        #val_fpn_meta_npz="./data/openimages_v7/yolo_world_fpn_P4_fp16_val_meta.npz",
        val_fpn_meta_npz="./data/openimages_v7/yolo_world_fpn_P4_fp16_train_meta.npz",
        classes=classes,
        device="cpu",
    )
    print(metrics)




if __name__ == "__main__":
    main()

    # import numpy as np
    # m = np.load("./data/openimages_v7/yolo_world_fpn_P4_fp16_train_meta.npz", allow_pickle=True)
    # print("shape:", m["shape"])   # (N,C,Hf,Wf)
    # print("stride:", int(m["stride"]))
    # Hf, Wf = m["shape"].tolist()[2:]
    # stride = int(m["stride"])
    # print("new_h,new_w:", Hf * stride, Wf * stride)