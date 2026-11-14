"""Cheap HW2 sanity baseline on provisional four-material data; test stays unopened.

This uses global RGB histograms, not pretrained CNN embeddings. It measures a lower
reference point, not material identification in real conditions or a seven-class MVP.
"""

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from waste_sorter.eda import file_sha256


def color_features(image: Image.Image) -> np.ndarray:
    pixels = np.asarray(image.convert("RGB").resize((64, 64)), dtype=float) / 255
    histograms = [
        np.histogram(pixels[:, :, channel], bins=16, range=(0, 1))[0] / 4096 for channel in range(3)
    ]
    return np.concatenate([*histograms, pixels.mean(axis=(0, 1)), pixels.std(axis=(0, 1))])


def run(data: Path, report: Path, configs: Path) -> dict:
    sources = json.loads((configs / "data_sources.json").read_text())
    archive = data / sources["trashnet"]["filename"]
    if file_sha256(archive) != sources["trashnet"]["sha256"]:
        raise ValueError("Archive checksum mismatch")
    manifest = pd.read_csv(report / "trashnet_manifest.csv")
    # Never load pixels from test. Remove repeated decoded images from these two subsets.
    selected = manifest[manifest.split.isin(["train", "validation"])].copy()
    selected = selected.drop_duplicates("pixel_sha256")
    if set(selected.split) != {"train", "validation"}:
        raise ValueError("Both train and validation are required")
    if selected.groupby("candidate_group").split.nunique().max() != 1:
        raise ValueError("Group leakage")
    features = []
    with ZipFile(archive) as source:
        for name in selected.archive_member:
            with source.open(name) as stream, Image.open(stream) as image:
                features.append(color_features(ImageOps.exif_transpose(image)))
    features = np.asarray(features)
    train = selected.split.eq("train").to_numpy()
    validation = selected.split.eq("validation").to_numpy()
    y = selected.target_label.to_numpy()
    classes = sorted(selected.target_label.unique())
    models = {
        "B0_most_frequent": DummyClassifier(strategy="most_frequent"),
        "B_color_logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42),
        ),
    }
    results = {
        "classes": classes,
        "train_images": int(train.sum()),
        "validation_images": int(validation.sum()),
        "test_evaluated": False,
        "manifest_sha256": file_sha256(report / "trashnet_manifest.csv"),
        "recipe": "RGB 16-bin histograms + channel mean/std (54 features), train-only scaler",
        "models": {},
    }
    predictions = selected.loc[validation, ["image_id", "target_label", "candidate_group"]].copy()
    for name, model in models.items():
        model.fit(features[train], y[train])
        predicted = model.predict(features[validation])
        predictions[name] = predicted
        results["models"][name] = {
            "macro_f1": float(f1_score(y[validation], predicted, labels=classes, average="macro")),
            "report": classification_report(
                y[validation], predicted, labels=classes, output_dict=True, zero_division=0
            ),
        }
        if name == "B_color_logistic":
            ConfusionMatrixDisplay.from_predictions(
                y[validation], predicted, labels=classes, cmap="Blues", colorbar=False
            )
            plt.title("Color baseline: validation only, provisional four-class task")
            plt.tight_layout()
            plt.savefig(report / "figures/09_baseline_validation.png", dpi=140)
            plt.close()
    (report / "baseline_metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    predictions.to_csv(report / "baseline_validation_predictions.csv", index=False)
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--report", type=Path, default=Path("reports/eda"))
    parser.add_argument("--configs", type=Path, default=Path("configs"))
    args = parser.parse_args()
    run(args.data_dir, args.report, args.configs)
