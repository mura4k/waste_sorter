"""Regression checks for data corruption, conservative grouping and leakage control."""

import io

import pandas as pd
import pytest
from PIL import Image

from waste_sorter.eda import assign_splits, audit_taco, group_similar_images, inspect_image


def image_bytes(color="red"):
    buffer = io.BytesIO()
    Image.new("RGB", (240, 240), color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_corrupt_image_is_reported_instead_of_silently_dropped():
    result = inspect_image(b"not an image")
    assert result["status"] == "decode_error"
    assert result["error"]
    assert result["sha256"]


def test_decode_reports_resolution_and_stable_pixel_identity():
    payload = image_bytes()
    result = inspect_image(payload)
    assert result["status"] == "ok"
    assert (result["width"], result["height"]) == (240, 240)
    assert not result["low_resolution"]
    assert result["pixel_sha256"] == inspect_image(payload)["pixel_sha256"]


def test_similar_images_are_grouped_even_when_labels_disagree():
    frame = pd.DataFrame(
        [
            {"image_id": "a", "source_label": "paper", **inspect_image(image_bytes())},
            {"image_id": "b", "source_label": "metal", **inspect_image(image_bytes())},
        ]
    )
    grouped, pairs = group_similar_images(frame)
    assert grouped.candidate_group.nunique() == 1
    assert pairs[0]["pixel_exact"]
    assert pairs[0]["label_conflict"]


def split_fixture():
    return pd.DataFrame(
        [
            {
                "image_id": f"{label}-{group}-{view}",
                "status": "ok",
                "target_label": label,
                "candidate_group": f"{label}-{group}",
            }
            for label in ["paper", "glass"]
            for group in range(20)
            for view in range(2)
        ]
    )


def test_split_keeps_views_together_and_is_reproducible():
    frame = split_fixture()
    first = assign_splits(frame)
    pd.testing.assert_frame_equal(first, assign_splits(frame))
    assert set(first.split) == {"train", "validation", "test"}
    assert first.groupby("candidate_group").split.nunique().eq(1).all()


def test_unresolved_and_cross_class_groups_are_held_out():
    frame = split_fixture()
    frame.loc[0, "target_label"] = "metal"
    frame.loc[2, "target_label"] = None
    result = assign_splits(frame)
    assert result.loc[:3, "split"].eq("excluded").all()
    assert result.loc[0, "exclusion_reason"] == "candidate_group_label_conflict"
    assert result.loc[2, "exclusion_reason"] == "unresolved_source_label"


def test_too_few_groups_does_not_fall_back_to_image_random_split():
    frame = split_fixture().iloc[:8]
    with pytest.raises(ValueError, match="Too few"):
        assign_splits(frame)


def test_taco_counts_scenes_separately_and_checks_bbox_and_references():
    document = {
        "images": [{"id": 1, "width": 100, "height": 100, "license": None}],
        "categories": [{"id": 1, "name": "Battery", "supercategory": "Battery"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [90, 90, 30, 30]},
            {"id": 3, "image_id": 999, "category_id": 1, "bbox": [0, 0, 10, 10]},
        ],
    }
    categories, _, result = audit_taco(document, {"hazardous": ["Battery"]})
    assert categories.iloc[0].annotations == 3
    assert result["stats"]["images"] == 1
    assert result["stats"]["invalid_references"] == 1
    assert result["stats"]["invalid_bboxes"] == 2
    assert result["stats"]["missing_license_images"] == 1


def test_taco_unknown_category_is_not_mapped_to_other():
    document = {
        "images": [{"id": 1, "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "Ambiguous composite", "supercategory": "Composite"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]}],
    }
    categories, _, _ = audit_taco(document, {})
    assert categories.iloc[0].provisional_target == "needs_review"


def test_modified_input_is_rejected_before_analysis(tmp_path):
    import json

    from waste_sorter.eda import run

    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "data_sources.json").write_text(
        json.dumps({"trashnet": {"filename": "changed.zip", "sha256": "not-matching"}})
    )
    (configs / "class_mapping.json").write_text("{}")
    (tmp_path / "changed.zip").write_bytes(b"modified archive")
    with pytest.raises(ValueError, match="checksum"):
        run(tmp_path, tmp_path / "output", configs)
    assert not (tmp_path / "output").exists()
