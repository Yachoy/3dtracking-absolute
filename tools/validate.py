"""CLI: запустить эксперимент и прогнать его через общий валидатор.

    python -m tools.validate --exp e00_oracle --data data --out out
    python -m tools.validate --exp e00_oracle --sigma-px 0 --no-video
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.dataset import load_scene  # noqa: E402
from core.validation import Validator  # noqa: E402
from experiments import EXPERIMENTS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", default="e00_oracle", choices=sorted(EXPERIMENTS))
    ap.add_argument("--data", default="data", help="каталог с распакованным архивом")
    ap.add_argument("--out", default="out", help="каталог для результатов")
    ap.add_argument("--sigma-px", type=float, default=1.5, help="шум 2D-наблюдений, px")
    ap.add_argument("--video-offset", type=int, default=0, help="сдвиг кадра видео относительно CSV")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-video", action="store_true", help="только метрики, без рендера")
    ap.add_argument("--pitch", action="store_true", help="наклонять бокс по уклону траектории")
    args = ap.parse_args()

    scene = load_scene(args.data)
    exp = EXPERIMENTS[args.exp](sigma_px=args.sigma_px)
    print(f"эксперимент {args.exp}: кадров GT {len(scene.target)}, камер {len(scene.cameras)}")
    pred = exp.run(scene)

    validator = Validator(
        scene,
        experiment_name=exp.name,
        video_offset=args.video_offset,
        out_dir=args.out,
        max_frames=args.max_frames,
        use_pitch=args.pitch,
    )
    validator.run(pred, render_video=not args.no_video)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
