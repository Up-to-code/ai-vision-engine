import json
import resource
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np

from .io import degradations, load_image
from .render import save_analysis


def benchmark(engine, directory, destination, degrade=False):
    directory, destination = Path(directory), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    images = (
        sorted((directory / "images").glob("*.jpg"))
        if (directory / "images").exists()
        else sorted(directory.glob("*.jpg"))
    )
    if not images:
        raise ValueError(f"No JPEG images in {directory}")
    rows, classes = [], Counter()
    start = perf_counter()
    for index, path in enumerate(images):
        image = load_image(path)
        variants = {"original": image}
        if degrade:
            variants.update(degradations(image))
        for kind, variant in variants.items():
            result = engine.analyze_image(variant, source=f"{path.name}:{kind}")
            folder = destination / path.stem / kind
            save_analysis(variant, result, engine.artifacts, folder)
            rows.append(
                {
                    "image": path.name,
                    "variant": kind,
                    "objects": len(result.objects),
                    "people": len(result.people),
                    "text_regions": len(result.text_regions),
                    "processing_ms": result.processing_ms,
                }
            )
            if kind == "original":
                classes.update(o.class_name for o in result.objects)
        print(f"Analyzed {index + 1}/{len(images)} {path.name}", flush=True)
    variants = {}
    for kind in sorted({r["variant"] for r in rows}):
        selected = [r for r in rows if r["variant"] == kind]
        times = [r["processing_ms"] for r in selected]
        variants[kind] = {
            "images": len(selected),
            "objects": sum(r["objects"] for r in selected),
            "median_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times, 95)),
        }
    import sys

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = peak / (1024**2 if sys.platform == "darwin" else 1024)
    report = {
        "images": len(images),
        "total_seconds": perf_counter() - start,
        "variants": variants,
        "classes": dict(classes),
        "peak_rss_mb": peak_mb,
        "model_initialization_ms": engine.manager.initialization_ms,
        "device": engine.manager.device,
        "model": engine.config.model,
        "note": "Counts and confidence are not accuracy. See holdout validation for AP.",
        "rows": rows,
    }
    (destination / "benchmark.json").write_text(json.dumps(report, indent=2))
    return report
