"""Create a reproducible static-image smoke video. Not a tracking accuracy dataset."""
import argparse
from pathlib import Path
import cv2

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('image', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('--frames', type=int, default=60)
args = parser.parse_args()
if args.output.exists():
    parser.error('Output already exists')
if args.frames <= 0:
    parser.error('--frames must be positive')
frame = cv2.imread(str(args.image))
if frame is None:
    parser.error('Cannot read image')
frame = cv2.resize(frame, (480, 640))
args.output.parent.mkdir(parents=True, exist_ok=True)
writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*'mp4v'), 15, (480, 640))
if not writer.isOpened():
    raise RuntimeError('Cannot create video')
try:
    for _ in range(args.frames):
        writer.write(frame)
finally:
    writer.release()
print(args.output)
