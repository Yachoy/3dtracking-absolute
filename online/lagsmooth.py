"""Сглаживание с фиксированной задержкой поверх покадрового трекера.

Для кадра t решается задача наименьших квадратов на окне t−history … t+lag:
    Σ (x_i − p_i)ᵀ W_i (x_i − p_i) + λ Σ ‖x_{i+1} − 2x_i + x_{i−1}‖²
где p_i — покадровая оценка трекера, W_i — её точность, λ — штраф за ускорение.

Точность анизотропная: если объект видит одна камера, направление на объект известно хорошо,
а дальность вдоль луча — плохо (дальность из размера силуэта). Поэтому вдоль луча вес меньше.
Выбросы гасятся весами Коши (IRLS). Ответ для кадра t готов на кадре t+lag;
lag=0 — строго онлайн (используются только прошлые кадры).

--pivot-forward сдвигает ответ вперёд по ходу движения: у трекера точка-ответ стоит примерно
на 0.85 м позади истинной (постоянно на всех участках Way1, это калибровка точки на кабине).

    python -m online.lagsmooth --replay out/base/replay.json --data data --lag 15 --reference data/....csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import block_diag, diags, identity, kron
from scipy.sparse.linalg import spsolve


def load_replay(path):
    frames = json.loads(Path(path).read_text(encoding="utf-8"))["frames"]
    xyz = np.array([np.array(f["xyz"], float) / 100 if f.get("xyz") else np.full(3, np.nan) for f in frames])
    return frames, xyz


def _read_ue_csv(path):
    df = pd.read_csv(path, sep=";", encoding="cp1251", dtype=str)
    cols = [c for c in ("Frame", "Loc_X", "Loc_Y", "Loc_Z") if c in df.columns]
    return df[cols].apply(lambda c: pd.to_numeric(c.str.replace(",", "."), errors="coerce"))


def camera_centers(data_dir, n_frames):
    """Центры камер (м) по кадрам: {номер камеры: (n_frames, 3)}."""
    out = {}
    for p in sorted(Path(data_dir).glob("object_transform_CineCameraActor*.csv")):
        c = _read_ue_csv(p)[["Loc_X", "Loc_Y", "Loc_Z"]].values / 100
        out[int(p.stem.rsplit("Actor", 1)[1])] = np.vstack([c, np.repeat(c[-1:], max(0, n_frames - len(c)), 0)])
    return out


def precision(frames, xyz, centers, sigma_multi, sigma_ray, sigma_depth_rel, sigma_none=25.0):
    """Матрица точности 3×3 для каждого кадра (0 — нет оценки)."""
    W = np.zeros((len(frames), 3, 3))
    for i, f in enumerate(frames):
        if np.isnan(xyz[i]).any():
            continue
        cams = f.get("cams") or []
        if len(cams) >= 2:
            W[i] = np.eye(3) / sigma_multi**2
        elif len(cams) == 1:
            r = xyz[i] - centers[cams[0]][f["frame"]]
            dist = np.linalg.norm(r)
            along = np.outer(r, r) / dist**2
            s_depth = max(sigma_multi, sigma_depth_rel * dist)
            W[i] = along / s_depth**2 + (np.eye(3) - along) / sigma_ray**2
        else:  # прогноз без наблюдения — почти не доверяем
            W[i] = np.eye(3) / sigma_none**2
    return W


def solve_window(p, W, lam, irls=2, cauchy_c=3.0):
    """Сглаженная траектория на окне. p: (n,3), W: (n,3,3)."""
    n = len(p)
    p = np.nan_to_num(p)
    if n < 3:
        return p
    D2 = diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n))
    S = lam * kron(D2.T @ D2, identity(3))
    w = np.ones(n)
    for _ in range(irls + 1):
        Wr = W * w[:, None, None]
        H = (block_diag(list(Wr)) + S).tocsc()
        x = spsolve(H, np.einsum("nij,nj->ni", Wr, p).ravel()).reshape(n, 3)
        d = x - p
        res = np.sqrt(np.maximum(np.einsum("ni,nij,nj->n", d, W, d), 0))
        w = 1 / (1 + (res / cauchy_c) ** 2)
    return x


def lag_smooth(xyz, W, lag, history=60, lam=30.0):
    """Ответ для каждого кадра t по окну t−history … t+lag (только то, что известно к кадру t+lag)."""
    n = len(xyz)
    out = np.full_like(xyz, np.nan)
    for t in range(n):
        a, b = max(0, t - history), min(n, t + lag + 1)
        if not W[a:b].any():
            continue
        out[t] = solve_window(xyz[a:b], W[a:b], lam)[t - a]
    return out


def shift_forward(xyz, meters, span=5, min_speed_mps=1.0, fps=30.0):
    """Сдвиг точки вперёд по курсу; курс — по уже выданным точкам (только прошлое: t−span … t)."""
    out = xyz.copy()
    if meters == 0:
        return out
    v = np.zeros_like(xyz)
    v[span:] = xyz[span:] - xyz[:-span]
    speed = np.linalg.norm(v[:, :2], axis=1) * fps / span
    ok = np.isfinite(speed) & (speed > min_speed_mps)
    out[ok, :2] += v[ok, :2] / np.linalg.norm(v[ok, :2], axis=1, keepdims=True) * meters
    return out


def evaluate(xyz, reference_csv):
    g = _read_ue_csv(reference_csv).dropna().astype({"Frame": int}).set_index("Frame") / 100
    idx = [i for i in g.index if i < len(xyz) and not np.isnan(xyz[i]).any()]
    e = np.linalg.norm(xyz[idx] - g.loc[idx].values, axis=1)
    return {"n": len(e), "mae_m": round(float(e.mean()), 3), "rmse_m": round(float(np.sqrt((e**2).mean())), 3),
            "median_m": round(float(np.median(e)), 3), "max_m": round(float(e.max()), 3),
            "lt_1m": round(float((e < 1).mean()), 3)}, pd.Series(e, index=idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--lag", type=int, default=0)
    ap.add_argument("--history", type=int, default=60)
    ap.add_argument("--lam", type=float, default=30000.0, help="штраф за ускорение (больше — глаже)")
    ap.add_argument("--sigma-multi", type=float, default=1.5)
    ap.add_argument("--sigma-ray", type=float, default=1.5)
    ap.add_argument("--sigma-depth-rel", type=float, default=0.02)
    ap.add_argument("--pivot-forward", type=float, default=0.0, help="сдвиг ответа вперёд по курсу, м")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--reference", type=Path)
    args = ap.parse_args()

    frames, xyz = load_replay(args.replay)
    W = precision(frames, xyz, camera_centers(args.data, len(frames)), args.sigma_multi, args.sigma_ray, args.sigma_depth_rel)
    sm = shift_forward(lag_smooth(xyz, W, args.lag, args.history, args.lam), args.pivot_forward)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("Frame;Loc_X;Loc_Y;Loc_Z;Available_At_Frame\n")
            for i, p in enumerate(sm):
                if not np.isnan(p).any():
                    f.write(f"{frames[i]['frame']};{p[0]*100:.4f};{p[1]*100:.4f};{p[2]*100:.4f};{min(i + args.lag, len(sm) - 1)}\n")
    if args.reference:
        print(json.dumps(evaluate(sm, args.reference)[0], ensure_ascii=False))


if __name__ == "__main__":
    main()
