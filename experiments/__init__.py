"""Реестр экспериментов: имя → класс, который отдаёт Prediction."""

from experiments.e00_oracle import OracleTriangulation
from experiments.e01_motion import MotionTriangulation
from experiments.e02_ekf import EkfMotion

EXPERIMENTS = {
    "e00_oracle": OracleTriangulation,
    "e01_motion": MotionTriangulation,
    "e02_ekf": EkfMotion,
}
