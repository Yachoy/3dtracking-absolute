"""Эксперименты-варианты поверх общего слоя `core`."""

from experiments.e00_oracle import OracleTriangulation
from experiments.e01_bgsub_ekf import BgSubEKF

EXPERIMENTS = {
    OracleTriangulation.name: OracleTriangulation,
    BgSubEKF.name: BgSubEKF,
}
