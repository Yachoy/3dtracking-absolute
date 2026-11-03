"""Сравнение нашей траектории с истинной: ошибки в метрах.

    python evaluate.py data/test out/trajectory.csv [--plot out/eval.png]
"""
import argparse

import numpy as np
import pandas as pd

from tracker.camera import CM, load_ground_truth, read_table


def evaluate(gt, pred):
    df = gt.join(pred.add_suffix("_pred")).dropna(subset=["Loc_X", "Loc_Y", "Loc_Z"])
    has_pred = df[["Loc_X_pred", "Loc_Y_pred", "Loc_Z_pred"]].notna().all(axis=1)
    d = df.loc[has_pred, ["Loc_X_pred", "Loc_Y_pred", "Loc_Z_pred"]].values - df.loc[has_pred, ["Loc_X", "Loc_Y", "Loc_Z"]].values
    err3d, err2d = np.linalg.norm(d, axis=1), np.linalg.norm(d[:, :2], axis=1)
    return {
        "кадров с GT": len(df),
        "покрытие, %": 100 * has_pred.mean(),
        "MAE 3D, м": err3d.mean(),
        "RMSE 3D, м": np.sqrt((err3d**2).mean()),
        "медиана 3D, м": np.median(err3d),
        "p90 3D, м": np.percentile(err3d, 90),
        "макс 3D, м": err3d.max(),
        "MAE XY, м": err2d.mean(),
        "MAE Z, м": np.abs(d[:, 2]).mean(),
    }, pd.Series(err3d, index=df.index[has_pred])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("pred_csv")
    ap.add_argument("--plot")
    args = ap.parse_args()

    gt = load_ground_truth(args.data_dir)
    pred = read_table(args.pred_csv)[["Loc_X", "Loc_Y", "Loc_Z"]] / CM
    metrics, err = evaluate(gt, pred)
    for k, v in metrics.items():
        print(f"{k:>15}: {v:.2f}" if isinstance(v, float) else f"{k:>15}: {v}")
    print("\nсредняя ошибка по отрезкам по 100 кадров, м:")
    print(err.groupby(err.index // 100 * 100).mean().round(2).to_string())

    if args.plot:
        import matplotlib.pyplot as plt
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6))
        a1.plot(gt.Loc_X, gt.Loc_Y, "k-", label="GT")
        a1.plot(pred.Loc_X, pred.Loc_Y, "r.", ms=2, label="наш трек")
        a1.set_aspect("equal"), a1.legend(), a1.set_title("Вид сверху, м")
        a2.plot(err.index, err.values), a2.set_xlabel("кадр"), a2.set_ylabel("ошибка 3D, м"), a2.grid()
        fig.tight_layout(), fig.savefig(args.plot, dpi=110)
        print("график:", args.plot)


if __name__ == "__main__":
    main()
