import numpy as np
import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.utils.openimages as fouo
from fiftyone import ViewField as F

from train import train_mlp_from_embeddings_and_labels, FSLDataset
from torch.utils.data import DataLoader


def boxable_classes():
    import pandas as pd

    # Boxable (~600)
    boxable_url = "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv"

    # “Full” class descriptions used by Open Images V7 point labels in FiftyOne’s codebase
    # (This is *much* larger than boxable)
    full_url = "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions.csv"

    boxable = pd.read_csv(boxable_url, header=None, names=["mid", "name"])
    full = pd.read_csv(full_url, header=None, names=["mid", "name"])

    boxable_names = set(boxable["name"].astype(str))
    non_boxable = full[~full["name"].astype(str).isin(boxable_names)]

    print("boxable:", len(boxable))
    print("full:", len(full))
    print("non-boxable examples:", non_boxable["name"].sample(25, random_state=0).tolist())



def _download_and_embed():
    seed = 51
    rng = np.random.default_rng(seed)


    open_images_v7_train = foz.load_zoo_dataset(
        "open-images-v7",
        split="train",
        label_types=["classifications"],
        # classes=picked_classes,     # optional: restrict pool to these classes
        max_samples=200000,          # candidate pool size
        shuffle=True,
        seed=seed,
        only_matching=True,         # optional but important for your filter semantics
    )

    oiv7_single_label_train = open_images_v7_train.match(
        F("positive_labels.classifications").length() == 1
    )

    print(len(oiv7_single_label_train))

    open_images_v7_val = foz.load_zoo_dataset(
        "open-images-v7",
        split="validation",
        label_types=["classifications"],
        # classes=picked_classes,     # optional: restrict pool to these classes
        max_samples=20000,          # candidate pool size
        shuffle=True,
        seed=seed,
        only_matching=True,         # optional but important for your filter semantics
    )

    oiv7_single_label_val = open_images_v7_val.match(
        F("positive_labels.classifications").length() == 1
    )

    print(len(oiv7_single_label_val))

    label_path = "positive_labels.classifications.label"

    train_classes = set(oiv7_single_label_train.distinct(label_path)) - {None}
    val_classes   = set(oiv7_single_label_val.distinct(label_path)) - {None}

    common_classes = sorted(train_classes & val_classes)
    print("common classes:", len(common_classes))

    oiv7_single_label_train = oiv7_single_label_train.match(
        F(label_path).contains(common_classes)
    )
    oiv7_single_label_val = oiv7_single_label_val.match(
        F(label_path).contains(common_classes)
    )

    print("filtered train:", len(oiv7_single_label_train))
    print("filtered val:  ", len(oiv7_single_label_val))

    # Stable mapping: common_classes is already sorted
    class_to_idx = {c: i for i, c in enumerate(common_classes)}

    # Save the class list so you can decode ints later
    np.save("./data/openimages_v7/common_classes_200000SingleLabel.npy",
            np.array(common_classes, dtype=object))

    def _flatten_single_label_per_sample(label_lists):
        # label_lists is like: [["Cat"], ["Dog"], None, ...] for label-list fields
        out = []
        for v in label_lists:
            if v is None:
                out.append(None)
            elif isinstance(v, (list, tuple)):
                out.append(v[0] if len(v) else None)
            else:
                out.append(v)
        return out

    train_labels = _flatten_single_label_per_sample(
        oiv7_single_label_train.values(label_path)
    )
    val_labels = _flatten_single_label_per_sample(
        oiv7_single_label_val.values(label_path)
    )

    assert all(l is not None for l in train_labels)
    assert all(l is not None for l in val_labels)

    train_y = np.array([class_to_idx[l] for l in train_labels], dtype=np.int32)
    val_y   = np.array([class_to_idx[l] for l in val_labels], dtype=np.int32)

    print("np.bincount(train_y))")
    print(np.bincount(train_y))
    print("np.bincount(val_y))")
    print(np.bincount(val_y))

    np.save("./data/openimages_v7/train_label_200000SingleLabel.npy", train_y)
    np.save("./data/openimages_v7/val_label_20000SingleLabel.npy", val_y)

    # (Optional but recommended) save sample IDs for alignment/debugging
    np.save("./data/openimages_v7/train_ids_200000SingleLabel.npy",
            np.array(oiv7_single_label_train.values("id"), dtype=object))
    np.save("./data/openimages_v7/val_ids_20000SingleLabel.npy",
            np.array(oiv7_single_label_val.values("id"), dtype=object))


    model = foz.load_zoo_model("clip-vit-base32-torch", classes=common_classes)

    train_embeddings = oiv7_single_label_train.compute_embeddings(model, batch_size=32, num_workers=4)
    val_embeddings = oiv7_single_label_val.compute_embeddings(model, batch_size=32, num_workers=4)

    np.save("./data/openimages_v7/clip_embeddings_train200000SingleLabel.npy", train_embeddings)
    np.save("./data/openimages_v7/clip_embeddings_val20000SingleLabel.npy", val_embeddings)


def _alignment(labels_a, labels_b, num_classes: int | None = None, eps: float = 1e-12) -> float:
    """
    Jensen–Shannon similarity between class distributions of two label arrays.

    Returns:
        float in [0, 1], where 1.0 means identical distributions.

    Notes:
        - Uses log base 2, so JS divergence is in [0, 1] (bits).
    """
    a = np.asarray(labels_a)
    b = np.asarray(labels_b)

    if a.size == 0 or b.size == 0:
        return 0.0

    if num_classes is None:
        num_classes = int(max(a.max(initial=0), b.max(initial=0))) + 1

    pa = np.bincount(a.astype(int), minlength=num_classes).astype(np.float64)
    pb = np.bincount(b.astype(int), minlength=num_classes).astype(np.float64)

    pa_sum = pa.sum()
    pb_sum = pb.sum()
    if pa_sum <= 0 or pb_sum <= 0:
        return 0.0

    pa /= pa_sum
    pb /= pb_sum
    m = 0.5 * (pa + pb)

    # KL(p || m) with safe masking (0 * log(0/.) treated as 0)
    def _kl(p, q):
        mask = p > 0
        return np.sum(p[mask] * np.log2((p[mask] + eps) / (q[mask] + eps)))

    js_div = 0.5 * _kl(pa, m) + 0.5 * _kl(pb, m)   # in [0, 1]
    js_sim = 1.0 - float(js_div)
    return max(0.0, min(1.0, js_sim))




def coreset_with_target_alignment(
    labels_base: np.ndarray,
    labels_val: np.ndarray,
    subset_size: float,
    target_alignment: float,
    *,
    num_classes: int | None = None,
    tol: float = 0.01,
    max_iter: int = 50,
    min_per_class: int = 1,
    seed: int = 0,
):
    """
    Select indices into `labels_base` of size `subset_size` whose class distribution
    has (approximately) the requested JS-alignment to `labels_val`.

    Args:
        labels_base: 1D int labels for the base/training pool (size N)
        labels_val:  1D int labels for validation set (any size)
        subset_size: fraction of N
        target_alignment: desired JS similarity in [0, 1]
        num_classes: total number of classes (recommended to pass explicitly, e.g. 335)
        tol: acceptable absolute error in alignment
        max_iter: iterations for binary search over the mixing parameter
        min_per_class: force at least this many from each class (clipped by caps)
        seed: RNG seed for shuffling/sampling

    Returns:
        idxs: 1D array of selected indices into `labels_base` (dtype int)
    """
    rng = np.random.default_rng(seed)

    labels_base = np.asarray(labels_base)
    labels_val = np.asarray(labels_val)

    if labels_base.ndim != 1 or labels_val.ndim != 1:
        raise ValueError("labels_base and labels_val must be 1D arrays")

    if not (0.0 <= target_alignment <= 1.0):
        raise ValueError("target_alignment must be in [0, 1]")

    N = labels_base.size
    k = int(round(subset_size * N))
    k = max(1, min(k, N))

    if num_classes is None:
        num_classes = int(max(labels_base.max(initial=0), labels_val.max(initial=0))) + 1

    # Caps: how many items available per class in base
    caps = np.bincount(labels_base.astype(int), minlength=num_classes).astype(int)

    # Group base indices by class (for sampling)
    idxs_by_class = [np.flatnonzero(labels_base == c) for c in range(num_classes)]
    for c in range(num_classes):
        rng.shuffle(idxs_by_class[c])

    # Validation distribution q
    q_counts = np.bincount(labels_val.astype(int), minlength=num_classes).astype(float)
    if q_counts.sum() <= 0:
        raise ValueError("labels_val appears empty or invalid")
    q = q_counts / q_counts.sum()

    # Base availability distribution r (only among classes with cap>0)
    cap_sum = caps.sum()
    if cap_sum <= 0:
        raise ValueError("labels_base appears empty or invalid")
    r = caps.astype(float) / float(cap_sum)

    # --- helpers ---
    def _counts_from_probs_with_caps(probs: np.ndarray, k: int, caps: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs, dtype=float)
        caps = np.asarray(caps, dtype=int)

        eligible = caps > 0
        probs = np.where(eligible, probs, 0.0)
        s = probs.sum()
        if s <= 0:
            return np.zeros_like(caps)

        probs = probs / s
        raw = probs * k

        base = np.floor(raw).astype(int)
        base = np.minimum(base, caps)

        remaining = k - int(base.sum())
        frac = raw - np.floor(raw)

        # deterministic-ish tie break via tiny jitter
        order = np.argsort(-(frac + rng.random(frac.shape) * 1e-12))

        i = 0
        while remaining > 0 and i < len(order):
            c = order[i]
            if base[c] < caps[c]:
                base[c] += 1
                remaining -= 1
            i += 1

        # If still short because caps bind, fill from any class with remaining capacity
        if remaining > 0:
            order2 = np.argsort(-(probs + rng.random(probs.shape) * 1e-12))
            j = 0
            while remaining > 0 and j < len(order2):
                c = order2[j]
                if base[c] < caps[c]:
                    base[c] += 1
                    remaining -= 1
                j += 1

        return base

    def _make_counts_for_t(t: float) -> np.ndarray:
        # target probs by mixing available base distribution and val distribution
        p = (1.0 - t) * r + t * q

        forced = np.zeros(num_classes, dtype=int)
        if min_per_class > 0:
            forced = np.minimum(min_per_class, caps)
        k_free = k - int(forced.sum())
        if k_free < 0:
            # can't satisfy min_per_class; just return forced clipped to k
            # (rare; only if k too small)
            forced = forced.copy()
            # trim forced down to k by removing from largest forced classes
            while forced.sum() > k:
                c = int(np.argmax(forced))
                forced[c] -= 1
            return forced

        caps_free = caps - forced
        counts_free = _counts_from_probs_with_caps(p, k_free, caps_free)
        return forced + counts_free

    def _materialize_indices(counts: np.ndarray) -> np.ndarray:
        chosen = []
        for c, take in enumerate(counts):
            take = int(take)
            if take <= 0:
                continue
            chosen.append(idxs_by_class[c][:take])
        out = np.concatenate(chosen) if chosen else np.array([], dtype=int)
        rng.shuffle(out)
        # Safety (due to any weirdness)
        if out.size > k:
            out = out[:k]
        return out

    def _score_for_t(t: float):
        counts = _make_counts_for_t(t)
        idxs = _materialize_indices(counts)
        score = _alignment(labels_base[idxs], labels_val, num_classes=num_classes)
        return score, idxs

    # --- find achievable range ---
    s0, idx0 = _score_for_t(0.0)  # more "base-like"
    s1, idx1 = _score_for_t(1.0)  # more "val-like" (subject to caps)
    lo_t, hi_t = 0.0, 1.0

    best = min([(abs(s0 - target_alignment), s0, idx0, 0.0),
                (abs(s1 - target_alignment), s1, idx1, 1.0)],
               key=lambda x: x[0])

    # If target outside achievable range, return closest endpoint
    s_min = min(s0, s1)
    s_max = max(s0, s1)
    if target_alignment <= s_min + tol:
        return best[2]
    if target_alignment >= s_max - tol:
        return best[2]

    # --- binary search on t ---
    # Assumes score generally moves monotonically with t; we still track best found.
    for _ in range(max_iter):
        mid = 0.5 * (lo_t + hi_t)
        sm, idxm = _score_for_t(mid)

        err = abs(sm - target_alignment)
        if err < best[0]:
            best = (err, sm, idxm, mid)
            if err <= tol:
                break

        # Decide which side to keep.
        # If score increases with t, this works; if not, best-tracking still protects you.
        if sm < target_alignment:
            lo_t = mid
        else:
            hi_t = mid

    return best[2]




def alignment_oracle():


    num_classes = 335

    labels_val = np.load("./data/openimages_v7/val_label_20000SingleLabel.npy")
    labels = np.load("./data/openimages_v7/train_label_200000SingleLabel.npy")

    embeddings = np.load("./data/openimages_v7/clip_embeddings_train200000SingleLabel.npy")
    embeddings_val = np.load("./data/openimages_v7/clip_embeddings_val20000SingleLabel.npy")   

    print("_alignment(labels, labels_val) =", _alignment(labels, labels_val, num_classes=num_classes))


    for alignment in [0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925]:
        
        accuracies = [] 
        alignments = []
        for _ in range(10):
            idxs = coreset_with_target_alignment(
                labels_base=labels,
                labels_val=labels_val,
                subset_size=0.05,
                num_classes=num_classes,
                target_alignment=alignment,
            )
            alignments.append(_alignment(labels[idxs], labels_val, num_classes=num_classes))

            embeddings_train = embeddings[idxs]
            labels_train = labels[idxs]

            _, trainer, _ = (
                    train_mlp_from_embeddings_and_labels(
                        embeddings_train, labels_train, num_classes=num_classes
                    )
                )
            _, val_acc = trainer.validate(
                DataLoader(
                    FSLDataset(embeddings_val, labels_val, device=trainer.device),
                    batch_size=64,
                )
            )
            accuracies.append(val_acc)

            
        print(f"alignment: {np.mean(alignments):.4f}")
        print(f"accuracy: {np.mean(accuracies):.4f}")
        print("")






if __name__ == "__main__":
    #_download_and_embed()

    alignment_oracle()