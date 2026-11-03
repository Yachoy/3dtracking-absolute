"""Эксперименты-варианты поверх общего слоя `core`."""

from experiments.e00_oracle import OracleTriangulation
from experiments.e01_motion import MotionTriangulation

EXPERIMENTS = {
    OracleTriangulation.name: OracleTriangulation,
    MotionTriangulation.name: MotionTriangulation,
}
