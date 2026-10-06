import argparse
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Sequential YOLO + ByteTrack video baseline")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; must not exist")
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--device", default="cpu", help="cpu, mps, or CUDA device such as 0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--fps", type=float, help="Override missing/incorrect CFR metadata")
    args = parser.parse_args()
    if args.imgsz < 32 or args.imgsz % 32:
        parser.error("--imgsz must be a positive multiple of 32")
    if not args.input.is_file():
        parser.error(f"Input video does not exist: {args.input}")
    if args.output_dir.exists():
        parser.error("--output-dir must not already exist")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    from .detection import YoloDetector
    from .tracking import ByteTrackAdapter
    from .pipeline import run_video
    started = time.perf_counter()
    detector = YoloDetector(args.weights, args.device, args.imgsz)
    setup_seconds = time.perf_counter() - started
    config = dict(weights=args.weights, device=args.device, imgsz=args.imgsz,
                  model_setup_seconds=setup_seconds, warmup=False,
                  classes=[0], detector_confidence=0.1)
    try:
        summary = run_video(args.input, args.output_dir, detector, ByteTrackAdapter,
                            args.max_frames, args.fps, config)
    except (ValueError, FileExistsError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
