"""Reproducible image audit, conservative split and TACO annotation-only audit.

Usage: python -m waste_sorter.eda
No training, augmentation, original image export or human label validation occurs here.
"""

import argparse
import hashlib
import importlib.metadata
import io
import json
import platform
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

from waste_sorter.classes import WasteClass

TARGET_CLASSES = [item.value for item in WasteClass]
SPLITS = ["train", "validation", "test"]


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inspect_image(payload: bytes) -> dict:
    """Decode full pixels; quality flags are triage signals, never automatic rejections."""
    result = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
    try:
        with Image.open(io.BytesIO(payload)) as raw:
            raw.load()
            result.update(
                width=raw.width,
                height=raw.height,
                mode=raw.mode,
                format=raw.format,
                exif_present=bool(raw.getexif()),
            )
            image = ImageOps.exif_transpose(raw).convert("RGB")
            pixels = np.asarray(image)
            result["pixel_sha256"] = hashlib.sha256(
                str(image.size).encode() + pixels.tobytes()
            ).hexdigest()
            gray = np.asarray(image.convert("L").resize((128, 128)), dtype=float)
            lap = (
                -4 * gray[1:-1, 1:-1]
                + gray[:-2, 1:-1]
                + gray[2:, 1:-1]
                + gray[1:-1, :-2]
                + gray[1:-1, 2:]
            )
            small = np.asarray(image.convert("L").resize((9, 8)), dtype=float)
            bits = small[:, 1:] > small[:, :-1]
            dhash = sum(int(bit) << i for i, bit in enumerate(bits.ravel()))
            result.update(
                status="ok",
                error="",
                aspect_ratio=image.width / image.height,
                brightness=float(gray.mean()),
                contrast=float(gray.std()),
                sharpness=float(lap.var()),
                dhash=f"{dhash:016x}",
                dark_fraction=float((gray < 10).mean()),
                bright_fraction=float((gray > 245).mean()),
                low_resolution=min(image.size) < 224,
                low_contrast=bool(gray.std() < 15),
                low_sharpness=bool(lap.var() < 50),
            )
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        result.update(status="decode_error", error=type(exc).__name__)
    return result


def audit_trashnet(archive: Path, mapping: dict) -> pd.DataFrame:
    rows = []
    with ZipFile(archive) as source:
        for name in sorted(source.namelist()):
            path = Path(name)
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            if not name.startswith("dataset-resized/") or path.name.startswith("."):
                continue
            label = path.parent.name
            if label not in mapping:
                raise ValueError(f"Unmapped source label: {label}")
            rows.append(
                {
                    "image_id": f"trashnet/{label}/{path.name}",
                    "archive_member": name,
                    "source": "trashnet",
                    "source_label": label,
                    "target_label": mapping[label],
                    "label_status": "source_label" if mapping[label] else "needs_review",
                    **inspect_image(source.read(name)),
                }
            )
    if not rows:
        raise ValueError("No images found in TrashNet archive")
    return pd.DataFrame(rows)


def group_similar_images(frame: pd.DataFrame, threshold: int = 4) -> tuple[pd.DataFrame, list]:
    """Keep exact and candidate near-duplicates together; do not call these object IDs.

    dHash is a heuristic: it misses viewpoint changes and can join unrelated images.
    Candidate grouping prevents known similarity edges crossing splits, not all leakage.
    """
    if not 0 <= threshold <= 64:
        raise ValueError("dHash threshold must be between 0 and 64")
    frame = frame.copy().reset_index(drop=True)
    parent = list(range(len(frame)))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def join(first, second):
        a, b = root(first), root(second)
        parent[max(a, b)] = min(a, b)

    pairs = []
    valid = frame.index[frame.status == "ok"].tolist()
    hashes = {i: int(frame.at[i, "dhash"], 16) for i in valid}
    for position, first in enumerate(valid):
        for second in valid[position + 1 :]:
            exact = frame.at[first, "pixel_sha256"] == frame.at[second, "pixel_sha256"]
            distance = (hashes[first] ^ hashes[second]).bit_count()
            if exact or distance <= threshold:
                join(first, second)
                pairs.append(
                    {
                        "first": frame.at[first, "image_id"],
                        "second": frame.at[second, "image_id"],
                        "distance": distance,
                        "pixel_exact": exact,
                        "label_conflict": (
                            frame.at[first, "source_label"] != frame.at[second, "source_label"]
                        ),
                    }
                )
    frame["candidate_group"] = [f"g{root(i):05d}" for i in range(len(frame))]
    return frame, pairs


def assign_splits(frame: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Stratify candidate groups; uncertain/conflicting groups are never silently relabeled."""
    frame = frame.copy()
    frame["split"] = "excluded"
    frame["exclusion_reason"] = ""
    conflict = frame.groupby("candidate_group").target_label.nunique().gt(1)
    eligible = (frame.status == "ok") & frame.target_label.notna()
    conflicts = frame.candidate_group.isin(conflict[conflict].index)
    frame.loc[frame.status != "ok", "exclusion_reason"] = "decode_error"
    frame.loc[frame.target_label.isna(), "exclusion_reason"] = "unresolved_source_label"
    frame.loc[conflicts, "exclusion_reason"] = "candidate_group_label_conflict"
    # If any member is unresolved or unreadable, do not allocate the rest of the group.
    bad_groups = set(frame.loc[~eligible | conflicts, "candidate_group"])
    eligible &= ~frame.candidate_group.isin(bad_groups)
    frame.loc[~eligible & frame.exclusion_reason.eq(""), "exclusion_reason"] = "group_on_hold"
    groups = frame.loc[eligible].groupby("candidate_group").target_label.first()
    if len(groups) == 0:
        return frame
    counts = groups.value_counts()
    if counts.min() < 7:
        raise ValueError("Too few independent candidate groups for stratified three-way split")
    train, holdout = train_test_split(
        groups.index.to_numpy(), test_size=0.30, random_state=seed, stratify=groups.values
    )
    validation, test = train_test_split(
        holdout, test_size=0.5, random_state=seed, stratify=groups.loc[holdout].values
    )
    for name, ids in zip(SPLITS, (train, validation, test), strict=True):
        frame.loc[frame.candidate_group.isin(ids), "split"] = name
    if frame.groupby("candidate_group").split.nunique().max() != 1:
        raise AssertionError("Candidate group crosses splits")
    return frame


def audit_taco(document: dict, mapping: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    categories = {category["id"]: category for category in document["categories"]}
    images = {image["id"]: image for image in document["images"]}
    if len(images) != len(document["images"]):
        raise ValueError("Duplicate TACO image IDs")
    if len(categories) != len(document["categories"]):
        raise ValueError("Duplicate TACO category IDs")
    labels = {name: target for target, names in mapping.items() for name in names}
    counts = Counter()
    scene_counts = Counter()
    scene_sets = {category: set() for category in categories}
    rows = []
    ids = set()
    for annotation in document["annotations"]:
        image = images.get(annotation["image_id"])
        category = categories.get(annotation["category_id"])
        x, y, width, height = annotation["bbox"]
        valid_bbox = bool(
            image
            and category
            and width > 0
            and height > 0
            and x >= 0
            and y >= 0
            and x + width <= image["width"] + 1
            and y + height <= image["height"] + 1
        )
        duplicate_id = annotation["id"] in ids
        ids.add(annotation["id"])
        rows.append(
            {
                "annotation_id": annotation["id"],
                "image_id": annotation["image_id"],
                "category_id": annotation["category_id"],
                "valid_reference": bool(image and category),
                "valid_bbox": valid_bbox,
                "duplicate_id": duplicate_id,
                "relative_bbox_area": (
                    width * height / (image["width"] * image["height"]) if image else None
                ),
            }
        )
        if category:
            counts[category["id"]] += 1
            scene_sets[category["id"]].add(annotation["image_id"])
        scene_counts[annotation["image_id"]] += 1
    category_rows = [
        {
            "category_id": key,
            "name": category["name"],
            "supercategory": category["supercategory"],
            "annotations": counts[key],
            "scenes": len(scene_sets[key]),
            "provisional_target": labels.get(category["name"], "needs_review"),
        }
        for key, category in categories.items()
    ]
    scenes = pd.DataFrame(
        [
            {
                "image_id": key,
                "width": image["width"],
                "height": image["height"],
                "annotations": scene_counts[key],
                "license_present": image.get("license") is not None,
            }
            for key, image in images.items()
        ]
    )
    annotations = pd.DataFrame(rows)
    stats = {
        "images": len(images),
        "annotations": len(rows),
        "categories": len(categories),
        "missing_license_images": int((~scenes.license_present).sum()),
        "unresolved_license_images": sum(
            image.get("license") not in {lic["id"] for lic in document.get("licenses", [])}
            for image in images.values()
        ),
        "invalid_references": int((~annotations.valid_reference).sum()),
        "invalid_bboxes": int((~annotations.valid_bbox).sum()),
        "duplicate_annotation_ids": int(annotations.duplicate_id.sum()),
        "multi_object_scenes": int(scenes.annotations.gt(1).sum()),
        "zero_annotation_scenes": int(scenes.annotations.eq(0).sum()),
        "small_bbox_fraction": float(annotations.relative_bbox_area.lt(0.01).mean()),
        "max_objects_per_scene": int(scenes.annotations.max()),
    }
    return pd.DataFrame(category_rows), annotations, {"stats": stats, "scenes": scenes}


def save_figure(output: Path, filename: str) -> None:
    plt.tight_layout()
    plt.savefig(output / filename, dpi=140, metadata={"Software": "waste-sorter HW2 EDA"})
    plt.close()


def plot_results(frame, pairs, taco_categories, taco_annotations, scenes, output):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = ["#168f83", "#e2a33b", "#527bb8", "#b16b8d", "#807aaa", "#b6b6b6"]
    source_counts = frame.source_label.value_counts().sort_index()
    source_counts.plot.bar(color=colors, figsize=(8, 4), rot=0)
    plt.ylabel("Images")
    plt.title("TrashNet: observed source labels")
    save_figure(output, "01_source_classes.png")

    mapped = frame.target_label.value_counts().reindex(TARGET_CLASSES, fill_value=0)
    mapped.plot.bar(color="#168f83", figsize=(9, 4), rot=20)
    plt.ylabel("Images with provisional mapped labels")
    plt.title("Seven-class coverage: missing classes are not zero-error classes")
    save_figure(output, "02_target_coverage.png")

    valid = frame.loc[frame.status == "ok"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for axis, column, title in zip(
        axes,
        ["bytes", "brightness", "sharpness"],
        ["JPEG size (bytes)", "Mean grayscale (0-255)", "Laplacian variance at 128 px"],
        strict=True,
    ):
        values = valid[column]
        if column == "sharpness":
            values = np.log10(1 + values)
            title = "log10(1 + " + title + ")"
        axis.hist(values, bins=35, color="#527bb8", edgecolor="white")
        axis.set_title(title)
        axis.set_ylabel("Images")
    save_figure(output, "03_quality_distributions.png")

    valid.boxplot(column="brightness", by="source_label", figsize=(8, 4), grid=False)
    plt.suptitle("")
    plt.title("Brightness by source label (possible nuisance cue)")
    plt.xlabel("")
    plt.ylabel("Mean grayscale")
    save_figure(output, "04_brightness_by_class.png")

    split_counts = (
        pd.crosstab(frame.target_label, frame.split)
        .reindex(index=TARGET_CLASSES, columns=SPLITS, fill_value=0)
        .fillna(0)
    )
    split_counts.plot.bar(stacked=True, color=colors[:3], figsize=(9, 4), rot=20)
    plt.ylabel("Images")
    plt.title("Provisional internal split after candidate grouping")
    save_figure(output, "05_internal_split.png")

    top = taco_categories.nlargest(18, "annotations").sort_values("annotations")
    plt.figure(figsize=(9, 6))
    plt.barh(top.name, top.annotations, color="#527bb8")
    plt.xlabel("Annotations (not independent images)")
    plt.title("TACO: 18 most frequent categories")
    save_figure(output, "06_taco_categories.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(
        scenes.annotations, bins=np.arange(0, scenes.annotations.max() + 2), color=colors[0]
    )
    axes[0].set_xlabel("Objects per scene")
    axes[0].set_ylabel("Scenes")
    areas = taco_annotations.relative_bbox_area.dropna().clip(lower=1e-8)
    axes[1].hist(np.log10(areas), bins=35, color=colors[2])
    axes[1].set_xlabel("log10(bbox area / scene area)")
    axes[1].set_ylabel("Annotations")
    fig.suptitle("TACO: scene context and object size")
    save_figure(output, "07_taco_scene_structure.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    group_sizes = frame.groupby("candidate_group").size()
    axes[0].hist(group_sizes, bins=np.arange(1, group_sizes.max() + 2), color=colors[0])
    axes[0].set_xlabel("Images per conservative candidate group")
    axes[0].set_ylabel("Groups")
    axes[1].hist([pair["distance"] for pair in pairs], bins=np.arange(-0.5, 5.5), color=colors[2])
    axes[1].set_xlabel("dHash distance in candidate pairs")
    axes[1].set_ylabel("Pairs")
    fig.suptitle("Similarity audit: candidates require manual review")
    save_figure(output, "08_similarity_candidates.png")


def run(data_dir: Path, output: Path, configs: Path, seed: int = 42) -> dict:
    sources = json.loads((configs / "data_sources.json").read_text())
    mapping = json.loads((configs / "class_mapping.json").read_text())
    for name, source in sources.items():
        if file_sha256(data_dir / source["filename"]) != source["sha256"]:
            raise ValueError(f"Input checksum mismatch: {name}")
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    frame = audit_trashnet(data_dir / sources["trashnet"]["filename"], mapping["trashnet"])
    frame, pairs = group_similar_images(frame)
    frame = assign_splits(frame, seed)
    frame.to_csv(output / "trashnet_manifest.csv", index=False, float_format="%.6f")
    pd.DataFrame(
        pairs, columns=["first", "second", "distance", "pixel_exact", "label_conflict"]
    ).to_csv(output / "similarity_pairs.csv", index=False)
    document = json.loads((data_dir / sources["taco"]["filename"]).read_text())
    categories, annotations, taco = audit_taco(document, mapping["taco"])
    categories.to_csv(output / "taco_categories.csv", index=False)
    annotations.to_csv(output / "taco_annotation_audit.csv", index=False, float_format="%.6f")
    taco["scenes"].to_csv(output / "taco_scenes.csv", index=False)
    valid = frame.loc[frame.status == "ok"]
    group_sizes = frame.groupby("candidate_group").size()
    exact_groups = valid.groupby("pixel_sha256").size()
    stats = {
        "seed": seed,
        "mapping_version": mapping["version"],
        "sources": sources,
        "environment": {
            "python": platform.python_version(),
            **{
                name: importlib.metadata.version(name)
                for name in ["numpy", "pandas", "pillow", "matplotlib", "scikit-learn"]
            },
        },
        "trashnet": {
            "images": len(frame),
            "source_counts": frame.source_label.value_counts().sort_index().to_dict(),
            "target_counts": frame.target_label.value_counts().to_dict(),
            "unresolved_labels": int(frame.target_label.isna().sum()),
            "decode_errors": int((frame.status != "ok").sum()),
            "raw_bytes": int(frame.bytes.sum()),
            "dimensions": valid.groupby(["width", "height"]).size().to_dict(),
            "modes": valid["mode"].value_counts().to_dict(),
            "exif_present": int(valid.exif_present.sum()),
            "low_resolution": int(valid.low_resolution.sum()),
            "low_contrast_flags": int(valid.low_contrast.sum()),
            "low_sharpness_flags": int(valid.low_sharpness.sum()),
            "brightness_median": float(valid.brightness.median()),
            "sharpness_median": float(valid.sharpness.median()),
            "exact_file_duplicates": int(valid.sha256.duplicated().sum()),
            "pixel_duplicate_groups": int(exact_groups.gt(1).sum()),
            "pixel_duplicate_excess": int((exact_groups - 1).clip(lower=0).sum()),
            "candidate_pairs": len(pairs),
            "candidate_cross_source_label_pairs": sum(pair["label_conflict"] for pair in pairs),
            "candidate_groups": int(len(group_sizes)),
            "candidate_multi_image_groups": int(group_sizes.gt(1).sum()),
            "split_counts": frame.split.value_counts().to_dict(),
            "excluded_reasons": frame.exclusion_reason.value_counts().to_dict(),
            "split_by_target": pd.crosstab(frame.target_label, frame.split).to_dict(),
            "group_leakage_count": int(
                frame.groupby("candidate_group").split.nunique().gt(1).sum()
            ),
        },
        "taco": taco["stats"],
    }
    stats["trashnet"]["dimensions"] = {
        f"{width}x{height}": count
        for (width, height), count in stats["trashnet"]["dimensions"].items()
    }
    (output / "summary.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    plot_results(frame, pairs, categories, annotations, taco["scenes"], figures)
    # The exact executable environment can be reproduced even if version ranges later move.
    (output / "environment.txt").write_text(
        "\n".join(
            f"{name}=={version}"
            for name, version in stats["environment"].items()
            if name != "python"
        )
        + "\n"
    )
    print(json.dumps({"trashnet": stats["trashnet"], "taco": stats["taco"]}, indent=2))
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("reports/eda"))
    parser.add_argument("--configs", type=Path, default=Path("configs"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.data_dir, args.output, args.configs, args.seed)
