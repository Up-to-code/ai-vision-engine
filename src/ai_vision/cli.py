import argparse
import json
from pathlib import Path
from time import monotonic

import cv2

from .config import VisionConfig, select_device
from .engine import VisionEngine
from .io import capture, load_image
from .render import overlay, save_analysis


def parser():
    p = argparse.ArgumentParser(description="AI Vision local perception engine")
    p.add_argument("--config", type=Path)
    p.add_argument("--model")
    p.add_argument("--compute", default=None, help="auto, cpu, mps, cuda:0")
    p.add_argument("--ocr", action="store_true")
    p.add_argument("--ocr-languages", nargs="+", default=None)
    for name in ("pose", "face", "depth", "segmentation"):
        p.add_argument(f"--{name}", action="store_true")
    commands = p.add_subparsers(dest="command", required=True)
    image = commands.add_parser("image")
    image.add_argument("source", type=Path)
    image.add_argument("--output", default="outputs/image")
    for name in ("video", "camera"):
        sub = commands.add_parser(name)
        if name == "video":
            sub.add_argument("source")
        else:
            sub.add_argument("--device", type=int, default=0)
        sub.add_argument("--output", default=f"outputs/{name}")
        sub.add_argument("--display", action="store_true")
        sub.add_argument("--max-frames", type=int)
    collect = commands.add_parser("collect")
    collect.add_argument("--count", type=int, default=100)
    collect.add_argument("--seed", type=int, default=20261010)
    collect.add_argument("--archive")
    collect.add_argument("--output", default="data/benchmarks/street_scenes")
    bench = commands.add_parser("benchmark")
    bench.add_argument("--source", default="data/benchmarks/street_scenes")
    bench.add_argument("--output", default="outputs/benchmark")
    bench.add_argument("--degrade", action="store_true")
    for name in ("train", "validate"):
        sub = commands.add_parser(name)
        sub.add_argument("--data", default="data/benchmarks/street_scenes/dataset.yaml")
        sub.add_argument("--output", default=f"runs/{name}")
        if name == "train":
            sub.add_argument("--epochs", type=int, default=10)
    film = commands.add_parser("film")
    film.add_argument("--source", default="outputs/benchmark")
    film.add_argument("--output", default="outputs/detections.mp4")
    film.add_argument("--seconds", type=float, default=50)
    export = commands.add_parser("export")
    export.add_argument("--format", choices=["onnx", "coreml"], default="onnx")
    return p


def run_stream(engine, source, destination, display=False, max_frames=None):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    engine.reset_tracking()
    writer = None
    try:
        with capture(source) as cap, (destination / "frames.jsonl").open("w") as stream:
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps = fps if 0 < fps < 240 else 30.0
            index, start = 0, monotonic()
            while max_frames is None or index < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                timestamp = monotonic() - start if isinstance(source, int) else index / fps
                result = engine.analyze_frame(
                    frame, timestamp_seconds=timestamp, frame_id=index, source=str(source)
                )
                view = overlay(frame, result)
                cv2.putText(
                    view,
                    f"{1000 / max(result.processing_ms, 0.001):.1f} pipeline FPS | "
                    f"{len(result.objects)} objects | {len(result.people)} people",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                if writer is None:
                    writer = cv2.VideoWriter(
                        str(destination / "tracked.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (frame.shape[1], frame.shape[0]),
                    )
                    if not writer.isOpened():
                        raise RuntimeError("Cannot create video writer")
                writer.write(view)
                stream.write(result.model_dump_json() + "\n")
                if display:
                    cv2.imshow("AI Vision - Q to exit", view)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                index += 1
            if index == 0:
                raise RuntimeError("Source yielded no readable frames")
    finally:
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()


def main():
    args = parser().parse_args()
    config = VisionConfig.load(args.config) if args.config else VisionConfig()
    updates = {}
    if args.model:
        updates["model"] = args.model
    if args.compute:
        updates["device"] = args.compute
    if args.ocr_languages:
        updates["ocr_languages"] = args.ocr_languages
    for name in ("ocr", "pose", "face", "depth", "segmentation"):
        if getattr(args, name):
            updates[name] = True
    config = config.model_copy(update=updates)
    if args.command == "collect":
        from .dataset import collect

        collect(args.output, args.count, args.seed, args.archive)
        return
    if args.command == "film":
        from .film import make_film

        print(make_film(args.source, args.output, args.seconds))
        return
    engine = VisionEngine(config)
    if args.command == "image":
        image = load_image(args.source)
        result = engine.analyze_image(image, source=str(args.source))
        save_analysis(image, result, engine.artifacts, args.output)
        print(result.model_dump_json(indent=2))
    elif args.command in ("camera", "video"):
        run_stream(
            engine,
            args.device if args.command == "camera" else args.source,
            args.output,
            args.display,
            args.max_frames,
        )
    elif args.command == "benchmark":
        from .benchmark import benchmark

        benchmark(engine, args.source, args.output, args.degrade)
    elif args.command in ("train", "validate"):
        model = engine.manager.yolo(config.model)
        options = dict(
            data=args.data,
            imgsz=config.image_size,
            device=select_device(config.device),
            batch=4,
            workers=0,
            project=args.output,
            name="run",
            exist_ok=True,
        )
        if args.command == "train":
            model.train(
                **options,
                epochs=args.epochs,
                seed=20261010,
                deterministic=True,
                freeze=10,
                lr0=0.001,
                optimizer="AdamW",
                plots=True,
            )
        else:
            metrics = model.val(**options)
            path = Path(args.output) / "metrics.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(metrics.results_dict, indent=2))
    elif args.command == "export":
        print(engine.manager.yolo(config.model).export(format=args.format, imgsz=config.image_size))


if __name__ == "__main__":
    main()
