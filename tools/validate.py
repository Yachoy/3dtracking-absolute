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


def _coerce(raw: str):
    """Строка из CLI -> bool / int / float / str."""
    low = raw.lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            pass
    return raw


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", default="e00_oracle", choices=sorted(EXPERIMENTS))
    ap.add_argument("--data", default="data", help="каталог с распакованным архивом")
    ap.add_argument("--out", default="out", help="каталог для результатов")
    ap.add_argument("--sigma-px", type=float, default=1.5, help="шум 2D-наблюдений, px")
    ap.add_argument("--video-offset", type=int, default=0, help="сдвиг кадра видео относительно CSV")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-video", action="store_true", help="только метрики, без рендера")
    ap.add_argument("--gt-points", action="store_true",
                    help="на камерах только GT-точка и выход трекера, без проекции модели")
    ap.add_argument("--smooth", type=int, default=15, help="окно сглаживания графика ошибки, кадров")
    ap.add_argument("--pitch", action="store_true", help="наклонять бокс по уклону траектории")
    ap.add_argument("--param", action="append", default=[], metavar="K=V",
                    help="параметр эксперимента (можно несколько раз)")
    ap.add_argument("--tag", default="", help="суффикс к имени эксперимента в out/")
    ap.add_argument("--workers", type=int, default=0,
                    help="процессов на камеры (0 — по числу камер и ядер)")
    args = ap.parse_args()

    scene = load_scene(args.data)
    kwargs = {}
    for item in args.param:
        key, _, raw = item.partition("=")
        kwargs[key.strip()] = _coerce(raw.strip())
    if args.workers:
        kwargs["workers"] = args.workers
    exp = EXPERIMENTS[args.exp](sigma_px=args.sigma_px, **kwargs)
    if args.tag:
        exp.name = f"{exp.name}-{args.tag}"
    print(f"эксперимент {args.exp}: кадров GT {len(scene.target)}, камер {len(scene.cameras)}")
    pred = exp.run(scene)

    validator = Validator(
        scene,
        experiment_name=exp.name,
        video_offset=args.video_offset,
        out_dir=args.out,
        max_frames=args.max_frames,
        use_pitch=args.pitch,
        gt_boxes=not args.gt_points,
        smooth=args.smooth,
    )
    validator.run(pred, render_video=not args.no_video)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
