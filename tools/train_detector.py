"""Дообучение YOLO-seg на разметке из tools/detection_labels.py.

Класс фуры остаётся 7 («truck» COCO), поэтому готовые веса подставляются в трекер без изменений:
    python -m online.run ... --global-seg runs/peleng/<name>/weights/best.pt

Обучать честно: сцена для проверки не должна попадать в обучение (например, обучение на Way1 с
эталоном, проверка на Way2 с псевдоразметкой, и наоборот). Дым и туман уже добавлены в датасет;
здесь включены обычные аугментации Ultralytics (яркость, масштаб, сдвиг, отражение) без мозаики
в конце, чтобы сеть видела цельные кадры.

    python tools/train_detector.py --data datasets/peleng/data.yaml --model models/yolo26l-seg.pt --name way1-train
"""
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="data.yaml из tools/detection_labels.py")
    ap.add_argument("--model", type=Path, default=Path("models/yolo26l-seg.pt"))
    ap.add_argument("--name", default="peleng")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--freeze", type=int, default=10, help="заморозить первые N слоёв (мало данных — меньше переобучения)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(str(args.model))
    model.train(
        data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, name=args.name,
        project="runs/peleng", freeze=args.freeze, device=args.device, patience=10,
        mosaic=0.5, close_mosaic=10, fliplr=0.5, degrees=0.0, translate=0.1, scale=0.5,
        hsv_h=0.01, hsv_s=0.4, hsv_v=0.4, single_cls=False, plots=True,
    )
    best = Path(model.trainer.best)
    print(f"лучшие веса: {best}")
    print(f"проверка: python tools/detection_labels.py recall --weights {best} ...")
    print(f"трекер:   python -m online.run ... --global-seg {best}")


if __name__ == "__main__":
    main()
