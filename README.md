# 3dtracking-absolute

Онлайн 3D-трекинг объекта по нескольким камерам с известными позами (Пеленг Space Hackathon 2.0).
Разбор задачи и план: [docs/razbor.md](docs/razbor.md).

## Установка

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install ultralytics scipy pandas opencv-python matplotlib imageio-ffmpeg
```

Данные кладём в `data/<набор>/` (видео `*.mp4` + `object_transform_*.csv`), папка в `.gitignore`.

## Запуск

```bash
python -m tracker.detect data/test                 # детекции YOLO26 -> cache/detections.csv
python run.py data/test --calibrate                # трекинг + подбор высоты объекта и сдвига по GT
python evaluate.py data/test out/trajectory.csv --plot out/eval.png
python -m tools.demo_video data/test --out out/demo.mp4  # демо: карта + 5 камер с рамками
```

На проверочном наборе (без GT):

```bash
python -m tracker.detect data/val --out cache/val_detections.csv
python run.py data/val --detections cache/val_detections.csv --out out/val_trajectory.csv
```

Отладка трекера без нейросети — синтетические детекции из GT:

```bash
python -m tools.fake_detections data/test
python run.py data/test --detections cache/fake_detections.csv --out out/fake.csv
```

## Как устроено

| Файл | Что делает |
|---|---|
| `tracker/camera.py` | чтение CSV из UE, модель камеры, проекция точки, луч из пикселя |
| `tracker/detect.py` | YOLO на каждом кадре каждой камеры (кадры независимы → онлайн) |
| `tracker/online.py` | 3D-трек: EKF по низу и высоте рамок всех камер, гейтинг, старт по паре камер или по одной |
| `run.py` | цикл по кадрам строго вперёд, запись `Frame;Loc_X;Loc_Y;Loc_Z` (см) |
| `evaluate.py` | ошибка в метрах vs GT |
| `tools/demo_video.py` | демо-видео: вид сверху (GT / трек) + 5 камер с рамками YOLO и выбранными трекером |

Ограничение: на кадре t используются только кадры ≤ t.
