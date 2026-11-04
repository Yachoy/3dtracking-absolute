# Передача контекста (для новой сессии Claude на ПК с GPU)

Хакатон Пеленг Space Hackathon 2.0: по 5 видео (Unreal, камеры с известными позами) выдать 3D-координаты
грузовика на каждом кадре. Проверочный набор — 4 ноября 20:00, сдача до 24:00 (до 06:00 5 ноября со штрафом).
Пользователь — веб-разработчик, в ML новичок: объяснять просто и по-русски.

## Где что лежит

| Ветка (github.com/Yachoy/3dtracking-absolute) | Что | MAE на Way1 |
|---|---|---|
| `best-online` | **основная**: код R1venDev (`codex/online-tracking`) + наше сглаживание `online/lagsmooth.py` + DirectML | см. ниже |
| `baseline-online-tracker` | наше раннее решение: YOLO + поток + 3D-модель по 4 краям рамки | 7.25 м |
| `dev` | Yachoy: YOLO в окне поиска + Калман | 6.83 м |
| AlokiN-N `ekf-multicam` | SAM 2 + подгонка модели + глобальное сглаживание (офлайн) | 2.31–2.51 м (заявлено) |

## Что выяснили

- Опубликованный результат R1venDev: **онлайн 2.53 м** (проверено нашим `evaluate.py` по их `replay.json`
  с https://plane.tail6ffd57.ts.net:8443). Он получен на GPU-сервере **с PicoPose** (`--recover-second-view`).
- **Без PicoPose на CPU** тот же код дал **5.88 м**: расходится с 20-го кадра, на кадрах 250–300 улетает на 35 м.
  Решение чувствительно к окружению. Логи: `out/base.log`, результат `out/base/replay.json` (не в git).
- **SAM на каждом кадре** (`--cadence 1`, CPU, без PicoPose): **3.78 м** (медиана 2.43, макс 21) вместо 5.88 —
  перенос масок потоком между вызовами SAM действительно копит ошибку. С `lagsmooth --lag 30 --pivot-forward 0.85`
  → 3.00 м (медиана 1.87). Результат `out/cad1/` (не в git).
- Предел геометрии: при точности силуэта 1 px средняя ошибка ~0.7 м; кадры 200–300 (одна дальняя камера) — ~3.4 м.
  На кадрах с 2+ камерами предел ~0.1 м, а реальная ошибка ~1.9 м — запас в качестве масок/подгонки.
- Ошибка — медленный дрейф (автокорреляция 0.89 на 1 кадре, 0.37 на 10).
- У ответа R1venDev постоянный сдвиг **≈ −0.85 м вдоль хода** (точка-ответ позади истинной).
- `online/lagsmooth.py` на их опубликованном `replay.json`: задержка 30 кадров + `--pivot-forward 0.85` → **1.64 м**
  (медиана 1.37). Строго онлайн сглаживание не помогает. Пропорции модели (×3.1) проверены — менять не нужно.
- В GT кадры 402 и 691 битые (буквы), кадры 0-based, кадр 1001 без видео.

## Что делать на ПК с GPU (1080 Ti)

1. `onnxruntime-gpu[cuda,cudnn]` (CUDA 12; Pascal не поддержан CUDA 13), иначе `onnxruntime-directml`.
   Проверить: в логе `SAM providers: ['CUDAExecutionProvider', ...]` или `Dml...`.
2. Модели: `models/yolo26n-seg.onnx`, `models/sam2/sam2.1_hiera_tiny.{encoder,decoder}.onnx` (ссылки в `tools/get_*.sh`).
3. Данные: `data/test/` (5 mp4 + 6 csv из project2.7z).
4. Прогон с SAM на каждом кадре (идёт и на CPU, результат `out/cad1`):
   `python -m online.run --data data/test --out out/gpu --sam models/sam2/sam2.1_hiera_tiny --threads 6 --cadence 1 --mono-depth --guard-flow-jumps`
5. `python -m online.evaluate --out out/gpu --reference data/test/object_transform_SM_AA_track.csv`
6. `python -m online.lagsmooth --replay out/gpu/replay.json --data data/test --lag 30 --pivot-forward 0.85 --reference data/test/object_transform_SM_AA_track.csv`

## Следующие шаги (по выгоде)

1. Попытаться воспроизвести 2.53 м: PicoPose (`--pico-source`, `--pico-checkpoint`, `--recover-second-view`) — нужен PyTorch на GPU.
2. Маски чище: SAM на каждом кадре, модель SAM крупнее (small), обрезка тени.
3. Для одной камеры — высота дороги (у R1venDev `online/estimation/road_height.py`).
4. Уточнить у организаторов, разрешена ли задержка выдачи (называть «задержка k кадров», не «будущее»).

## Прогоны на ПК с GPU (1080 Ti, Windows, 4 ноября)

Окружение: Python 3.12, `onnxruntime-gpu[cuda,cudnn]==1.22.0` + `nvidia-cudnn-cu12==9.8.0.87`
(cuDNN 9.27 падает на Pascal и молча уходит на CPU), `torch` 2.14 cu126 (cu128 не поддерживает sm_61),
`einops omegaconf mmcv-lite timm`. PicoPose — `vendor/PicoPose` (коммит 543f8fe), чекпойнт `vendor/picopose.ckpt`.
Полная сетка для `--appearance-target`: `python tools/truck_mesh.py CargoTruck.fbx models/truck`
(без Assimp, в системе оболочек `truck_hulls.json`, ×3.1).

| Прогон | MAE онлайн | Медиана | Макс | lagsmooth 30 + 0.85 |
|---|---|---|---|---|
| без PicoPose, `--cadence 1 --mono-depth --guard-flow-jumps` | 3.78 | 2.43 | 21.2 | 3.00 |
| **PicoPose, `--mono-depth --recover-second-view`** | **3.01** | 1.99 | 16.0 | **2.00** |
| PicoPose + `--cadence 1 --guard-flow-jumps` | 3.57 | 2.30 | 21.2 | 2.85 |

Без PicoPose GPU даёт те же 3.78 м, что CPU. С PicoPose ~10 мин на 1001 кадр. `recovered_camera` не принят
ни разу (у R1venDev — кадр 512): вероятная причина разрыва с 2.53 м — их сетка/версии.
