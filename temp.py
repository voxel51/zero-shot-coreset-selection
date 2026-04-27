import fiftyone as fo
import fiftyone.zoo as foz
import numpy as np
# 1) Text embeddings (compute once)
prompts = [f"{model.config.text_prompt} {c}" for c in common_classes]
text_emb = model.embed_prompts(prompts).astype(np.float32)           # (C, D)
text_emb /= np.linalg.norm(text_emb, axis=1, keepdims=True)

# CLIP logit scale (matches FiftyOne's TorchCLIPModel implementation)
logit_scale = float(model._model.logit_scale.exp().detach().cpu().item())

def _zs_top1_and_true_conf(image_emb: np.ndarray, true_y: np.ndarray | None, chunk: int = 4096):
    img = image_emb.astype(np.float32)                               # (N, D)
    img /= np.linalg.norm(img, axis=1, keepdims=True)

    n = img.shape[0]
    pred_idx = np.empty(n, dtype=np.int32)
    pred_conf = np.empty(n, dtype=np.float32)
    true_conf = np.empty(n, dtype=np.float32) if true_y is not None else None

    for i in range(0, n, chunk):
        logits = logit_scale * (img[i : i + chunk] @ text_emb.T)      # (B, C)

        # stable softmax
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)

        pi = probs.argmax(axis=1).astype(np.int32)
        pred_idx[i : i + chunk] = pi
        pred_conf[i : i + chunk] = probs[np.arange(pi.size), pi]

        if true_y is not None:
            yi = true_y[i : i + chunk].astype(np.int32)
            true_conf[i : i + chunk] = probs[np.arange(yi.size), yi]

    return pred_idx, pred_conf, true_conf

train_pred_idx, train_pred_conf, train_true_conf = _zs_top1_and_true_conf(train_embeddings, train_y)
val_pred_idx,   val_pred_conf,   val_true_conf   = _zs_top1_and_true_conf(val_embeddings,   val_y)

np.save("./data/openimages_v7/zs_pred_idx_train200000SingleLabel.npy", train_pred_idx)
np.save("./data/openimages_v7/zs_pred_conf_train200000SingleLabel.npy", train_pred_conf)
np.save("./data/openimages_v7/zs_true_conf_train200000SingleLabel.npy", train_true_conf)
