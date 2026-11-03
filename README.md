# 3dtracking-absolute

Онлайн 3D-трекинг объекта по нескольким камерам с известными позами (Пеленг Space Hackathon 2.0).
Разбор задачи и план: [docs/razbor.md](docs/razbor.md).

## Установка

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install ultralytics scipy pandas opencv-python matplotlib
```

Данные кладём в `data/<набор>/` (видео `*.mp4` + `object_transform_*.csv`), папка в `.gitignore`.

## Запуск

```bash
python -m tracker.detect data/test                 # детекции YOLO26 -> cache/detections.csv
python run.py data/test --calibrate                # трекинг + подбор сдвига до точки-ответа по GT
python evaluate.py data/test out/trajectory.csv --plot out/eval.png
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
| `tracker/online.py` | 3D-трек: EKF по пикселям всех камер, гейтинг, инициализация по паре камер |
| `run.py` | цикл по кадрам строго вперёд, запись `Frame;Loc_X;Loc_Y;Loc_Z` (см) |
| `evaluate.py` | ошибка в метрах vs GT |

Ограничение: на кадре t используются только кадры ≤ t.
