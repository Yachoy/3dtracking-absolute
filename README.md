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

3D-модель объекта (FBX от организаторов) — трекер сравнивает с рамкой YOLO все 4 края спроецированной модели,
точка-ответ (перед кабины) берётся из модели. Подключается автоматически, если есть `config/truck_model.json`:

```bash
python -m tracker.truck_model data/CargoTruck.fbx --fit data/test   # -> config/truck_model.json (масштаб и точка-ответ по GT)
python run.py data/test --no-model                                   # сравнить без модели
```

Заполнение пропусков YOLO оптическим потоком (2D-трекер вперёд и назад от рамок YOLO):

```bash
python -m tracker.flow_fill data/test              # -> cache/flow_boxes.csv (~40 с)
python run.py data/test --flow                     # онлайн: + рамки потока вперёд
python run.py data/test --flow --replay            # онлайн: + пересчёт трека по обратному проходу
                                                   #   (+ out/trajectory_delayed.csv — с задержкой, не строгий онлайн)
python -m tools.demo_video data/test --flow --out out/demo_flow.mp4
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
| `tracker/truck_model.py` | чтение FBX, оболочка модели, подгонка масштаба и точки-ответа, проекция рамки |
| `tracker/flow_fill.py` | 2D-трекер по оптическому потоку между рамками YOLO (с вычетом поворота камеры) |
| `tracker/online.py` | 3D-трек: EKF по низу и высоте рамок всех камер, гейтинг, старт по паре камер или по одной |
| `run.py` | цикл по кадрам строго вперёд, запись `Frame;Loc_X;Loc_Y;Loc_Z` (см) |
| `evaluate.py` | ошибка в метрах vs GT |
| `tools/demo_video.py` | демо-видео: вид сверху (GT / трек) + 5 камер с рамками YOLO и выбранными трекером |

Ограничение: на кадре t используются только кадры ≤ t.
