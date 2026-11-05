# Почему фура видна, а ответа нет: опции захвата и дообучение детектора

Ветка `way2-no-gaps`. Заполнение провалов (`NO_GAPS.md`, v5) закрывает пустые кадры прогнозом, но
дальше точность упирается в трекер и детектор. На Way1 v5 даёт MAE 3,91 м против 2,46 м у демо на
старом трекере, и разница почти целиком — кадры, где новый трекер не взял фуру, хотя она видна.

## Что мешает (проверено по эталону Way1)

| Кадры Way1 | Что видно | Почему нет ответа |
|---|---|---|
| 0–99 | фуру видят камеры 1 и 2; YOLO26l находит её в обеих с уверенностью ≥ 0,3 (`tools/detection_labels.py recall`, кадры 80–81) | в камере 2 рамку режет нижний край кадра, а захват `ConsensusTracker.acquire` обрезанные маски не берёт — пары нет |
| 232–312 | фуру видит только камера 2, остальные закрыты горой и деревьями | захват требует две камеры |
| 383–411 | маски SAM2 в камерах 2, 4, 5 | трекер ведёт прогнозом и уезжает на 44 м (это теперь подхватывает заполнение v5) |

На Way2 добавляется сам детектор: в пыли и дыму уверенность YOLO на белой фуре падает до 0,05–0,15,
сверху (камера 2, дрон) он фуру почти не узнаёт.

## 1. Опции захвата (выключены по умолчанию — без флагов трекер прежний)

- `--acquire-clipped` — захват по паре камер, где одна маска обрезана краем кадра (обе обрезанные —
  нельзя). Тест: `tests/test_acquire_options.py` — силуэт по эталону Way1, кадр 80: без флага захвата
  нет, с флагом — по камерам 1 и 2 с ошибкой < 3 м.
- `--single-camera` — захват по одной камере в многокамерной сцене, когда пары нет: уверенная
  детекция (≥ 0,5, не обрезана, силуэт CAD ≥ 0,68) держится ≥ 5 кадров подряд в одной камере,
  движется не быстрее машины, место внутри зоны камер и не глубоко под ними (так отсеивается огонь
  Way2), и недалеко от последней позы с наблюдением.

Проверка (Way1 — по эталону, Way2 — покрытие и скачки):

```bash
python -m online.run --data data/test --out out/w1-base --global-seg models/yolo26l-seg.pt --sam-video facebook/sam2.1-hiera-base-plus --no-fill-gaps
python -m online.run --data data/test --out out/w1-acq  --global-seg models/yolo26l-seg.pt --sam-video facebook/sam2.1-hiera-base-plus --no-fill-gaps --acquire-clipped --single-camera
python -m online.evaluate --out out/w1-acq --reference data/test/object_transform_SM_AA_track.csv
```

То же для `data/way2`. Сравнить: покрытие трекера, MAE на Way1, шаги > 10 м и засечки на Way2;
затем — вместе с заполнением (без `--no-fill-gaps`). Флаги можно проверять и по одному.

## 2. Полнота детекции: где фура видна, но YOLO её не находит

`tools/detection_labels.py recall` ставит CAD в позу по эталону (или по кадрам `multi` трекера) и
проецирует в камеры. Закрытие горой или деревом проекция не знает, поэтому видимость проверяет SAM2
по рамке проекции: маска должна совпасть с силуэтом по форме (IoU ≥ 0,55), уверенность SAM2 ≥ 0,8,
и под маской не должно быть листвы (средний лапласиан ≤ 55: у листвы 67–137, у борта фуры 11–40).
Для видимых фур запускается YOLO как в `--global-seg` (весь кадр, 960) и по окну вокруг фуры (640).

```bash
python tools/detection_labels.py recall --data data/test --reference data/test/object_transform_SM_AA_track.csv \
    --sam models/sam2/sam2.1_hiera_tiny --weights models/yolo26l-seg.pt --out out/recall-way1
python tools/detection_labels.py recall --data data/way2 --replay out/w2-base/replay.json \
    --sam models/sam2/sam2.1_hiera_tiny --weights models/yolo26l-seg.pt --out out/recall-way2
```

Печатает долю найденных при порогах 0,05 / 0,16 / 0,30 по камерам, размерам и обрезанным краем;
вырезки пропусков — в `out/recall-*/misses`.

## 3. Дообучение YOLO на наших сценах

Разметка автоматическая: маска = SAM2 ∩ силуэт CAD, класс 7 («truck» COCO), чтобы веса подходили к
`--global-seg` без изменений. Часть обучающих кадров дублируется с синтетическим дымом/туманом и
пониженным контрастом. Обучать и проверять на разных сценах:

```bash
python tools/detection_labels.py dataset --data data/test --reference data/test/object_transform_SM_AA_track.csv \
    --sam models/sam2/sam2.1_hiera_tiny --out datasets/peleng --split train --name way1
python tools/detection_labels.py dataset --data data/way2 --replay out/w2-base/replay.json \
    --sam models/sam2/sam2.1_hiera_tiny --out datasets/peleng --split val --name way2
python tools/train_detector.py --data datasets/peleng/data.yaml --model models/yolo26l-seg.pt --name way1-train
```

Потом — `recall` с новыми весами на Way2 и прогон трекера с `--global-seg runs/peleng/way1-train/weights/best.pt`.
Для честной проверки на Way1 — наоборот: обучение на Way2, проверка на Way1 по эталону.

Проверено на этом ПК только на нескольких кадрах (без полного прогона): разметка Way1, кадры 80, 250,
290, 415 — фура за деревом в камере 4 отсеяна, видимые фуры в камерах 1, 2, 3, 5 размечены; `recall`
на кадрах 80–81 — 4 видимые фуры, все найдены.
