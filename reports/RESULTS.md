# Executed results — 10 October 2026

100 licensed COCO internet images, 80 train / 20 holdout. Detection benchmark: 700 actual inferences across originals and six controlled degradation variants. All optional modules ran on each of the 100 originals. No camera was used.

## Detection validation

| Model | mAP50 | mAP50–95 | Precision | Recall |
|---|---:|---:|---:|---:|
| Pretrained YOLO26 Nano | 0.6689 | 0.4720 | 0.7756 | 0.5715 |
| Fine-tuned, 3 epochs | 0.5780 | 0.3880 | 0.6149 | 0.5363 |

Fine-tuning lowered heldout AP. The pretrained detector remains the default. This 20-image holdout is from COCO validation; upstream development already used COCO. It is too small and insufficiently independent to claim production generalization. Crowd annotations are excluded from the converted YOLO labels.

## Controlled degradation

| Variant | Images | Detections | Median latency (ms) | P95 (ms) |
|---|---:|---:|---:|---:|
| dark | 100 | 650 | 26.42 | 35.65 |
| gaussian | 100 | 374 | 26.26 | 39.68 |
| jpeg | 100 | 385 | 26.57 | 35.89 |
| low_resolution | 100 | 412 | 26.68 | 33.01 |
| motion | 100 | 231 | 26.74 | 34.24 |
| noise | 100 | 508 | 26.72 | 43.48 |
| original | 100 | 699 | 26.78 | 43.52 |

Detector benchmark used cpu; elapsed 27.79 s including output writing; peak process RSS 444.0 MiB. Counts are not accuracy and fewer boxes can mean missed detections or removal of false positives.

## Optional modules

Full pipeline: CPU, 100 images, median 819.1 ms, P95 1127.7 ms; elapsed 88.0 seconds; peak RSS 3060.4 MiB. Model initialization and first inference are included in the first image; steady-state median excludes their influence. The initial smoke run included network downloads and took about 95 seconds, so it is not an inference-speed benchmark.

OCR: 72 text regions above confidence threshold; no OCR ground truth exists for this subset. Pose: 198 pose person outputs. Faces: 5 face-landmark outputs. Unknown expressions/actions are retained. Depth and segmentation outputs exist for all 100 images; their accuracy is not quantified without appropriate ground truth.

## Tracking verification

A deterministic 30-frame fixture translated one real photograph across image coordinates. ByteTrack produced 509 tracked boxes over 30 frames. This verifies temporal bookkeeping, not real-world tracking accuracy or human movement. Its synthetic displacement is explicitly not a natural video or a robot trajectory.

## Export comparison

The same first 20 images were measured after warm-up on CPU. Other benchmark jobs were running, so timing is indicative, not an isolated hardware benchmark.

| Model | Disk MiB | Median ms | P95 ms | Boxes |
|---|---:|---:|---:|---:|
| models/yolo26n.pt | 5.29 | 38.51 | 62.31 | 147 |
| models/yolo26n.onnx | 9.43 | 23.72 | 34.03 | 162 |

ONNX export succeeded, but output counts differ from PyTorch. It is offered as an optional export, not adopted as an accuracy-equivalent default. FP16, INT8, and Core ML were not benchmarked.

## Qualitative inspection

Inspected contact sheets and separate object, depth, and segmentation outputs. The sample includes clear streets, night traffic lights, rainy streets, distant objects, and transport interiors. Some images are not street scenes because the category filter admits vehicles indoors or on posters. In the initial contact sheet, image 000000125778 contains an indoor scene with a street poster; this is a useful false-context case, not evidence of a physical walkable path. Bounding boxes crowd together in dense scenes. Semantic labels and OCR can be wrong, particularly for tiny or blurry regions. No Arabic-specific street photo or verified naturally motion-blurred subset was guaranteed by collection; controlled blur is explicitly synthetic.

## Validation performed

12 pytest tests passed, including actual pretrained inference. Ruff checks and formatting checks passed. Camera lifecycle was tested with mocks only. Optional modules ran on real inputs and no outputs were fabricated. A 50-second, 1280×720 H.264 film contains all 100 analyzed originals with animated overlays, using the better pretrained detector. The animation is not tracking across independent photographs.
