"""Reproducible licensed COCO subset, real annotations, explicit train/holdout split."""

import hashlib
import json
import random
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import cv2
import yaml

ANNOTATIONS_URL = (
    "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip"
)


def collect(destination="data/benchmarks/street_scenes", count=100, seed=20261010, archive=None):
    if count < 2:
        raise ValueError("At least two images are required for train/validation splits")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    archive = Path(archive or "data/annotations.zip")
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(ANNOTATIONS_URL, archive)
    with zipfile.ZipFile(archive) as z:
        data = json.loads(z.read("annotations/instances_val2017.json"))
    licenses = {i["id"]: i for i in data["licenses"]}
    categories = sorted(data["categories"], key=lambda c: c["id"])
    mapping = {c["id"]: i for i, c in enumerate(categories)}
    annotations = defaultdict(list)
    for item in data["annotations"]:
        annotations[item["image_id"]].append(item)
    # Require a street-context category; this is a category filter, not a claim of scene weather.
    street = {2, 3, 4, 6, 8, 10, 13}  # bicycle/car/motorcycle/bus/truck/light/stop sign
    candidates = [
        im
        for im in data["images"]
        if im["license"] in {4, 5, 8}
        and any(a["category_id"] in street for a in annotations[im["id"]])
    ]
    random.Random(seed).shuffle(candidates)
    manifest = []
    failures = []

    def fetch(im):
        url = im["coco_url"].replace(
            "http://images.cocodataset.org/", "https://s3.amazonaws.com/images.cocodataset.org/"
        )
        target = destination / "images" / im["file_name"]
        target.parent.mkdir(exist_ok=True)
        try:
            if not target.exists():
                with urllib.request.urlopen(url, timeout=45) as response:
                    target.write_bytes(response.read())
            image = cv2.imread(str(target))
            if image is None or image.shape[:2] != (im["height"], im["width"]):
                raise ValueError("Downloaded image resolution/decode mismatch")
            license = licenses[im["license"]]
            return {
                **im,
                "download_url": url,
                "source": "COCO 2017 validation / Flickr",
                "license": license,
                "downloaded_at": datetime.now(UTC).isoformat(),
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        except (OSError, ValueError) as exc:
            target.unlink(missing_ok=True)
            return {"failed_id": im["id"], "error": str(exc)}

    # Download only enough candidates, retry replacements without changing successful samples.
    for offset in range(0, len(candidates), count):
        needed = count - len(manifest)
        if not needed:
            break
        with ThreadPoolExecutor(max_workers=6) as pool:
            for item in pool.map(fetch, candidates[offset : offset + needed]):
                if "failed_id" in item:
                    failures.append(item)
                else:
                    manifest.append(item)
    if len(manifest) != count:
        raise RuntimeError(f"Only {len(manifest)}/{count} images downloaded: {failures}")
    ids = {im["id"] for im in manifest}
    subset = {
        **data,
        "images": [im for im in data["images"] if im["id"] in ids],
        "annotations": [a for a in data["annotations"] if a["image_id"] in ids],
    }
    (destination / "annotations.json").write_text(json.dumps(subset))
    (destination / "manifest.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "images": manifest,
                "failures": failures,
                "annotation_source": ANNOTATIONS_URL,
            },
            indent=2,
        )
    )
    labels = destination / "labels"
    labels.mkdir(exist_ok=True)
    for im in manifest:
        lines = []
        for a in annotations[im["id"]]:
            if a.get("iscrowd"):
                continue
            x, y, w, h = a["bbox"]
            lines.append(
                f"{mapping[a['category_id']]} {(x + w / 2) / im['width']:.8f} "
                f"{(y + h / 2) / im['height']:.8f} {w / im['width']:.8f} {h / im['height']:.8f}"
            )
        (labels / Path(im["file_name"]).with_suffix(".txt")).write_text("\n".join(lines))
    cut = int(count * 0.8)
    for split, items in (("train", manifest[:cut]), ("val", manifest[cut:])):
        (destination / f"{split}.txt").write_text(
            "\n".join(str((destination / "images" / i["file_name"]).resolve()) for i in items)
            + "\n"
        )
    (destination / "dataset.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(destination.resolve()),
                "train": "train.txt",
                "val": "val.txt",
                "names": {i: c["name"] for i, c in enumerate(categories)},
            }
        )
    )
    print(f"Collected {count} licensed images: {cut} train / {count - cut} holdout", flush=True)
    return manifest
