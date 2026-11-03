"""Адаптер core.dataset.CameraTrack -> модель камеры для детектора и трекера.

Матрица поворота — как FRotationMatrix в UE (X вперёд, Y вправо, Z вверх).
Система камеры: x вправо, y вниз, z вперёд.
"""
from __future__ import annotations

import numpy as np


def ue_axes(roll, pitch, yaw):
    R, P, Y = np.radians([roll, pitch, yaw])
    SR, CR, SP, CP, SY, CY = np.sin(R), np.cos(R), np.sin(P), np.cos(P), np.sin(Y), np.cos(Y)
    fwd = np.array([CP*CY, CP*SY, SP])
    right = np.array([SR*SP*CY - CR*SY, SR*SP*SY + CR*CY, -SR*CP])
    up = np.array([-(CR*SP*CY + SR*SY), CY*SR - CR*SP*SY, CR*CP])
    return fwd, right, up


class TrackCamera:
    def __init__(self, track, width=1920, height=1080):
        self.track = track
        self.focal = track.focal_mm
        self.fx = width * track.focal_mm / track.sensor_mm[0]
        self.fy = height * track.focal_mm / track.sensor_mm[1]
        self.cx, self.cy = width / 2, height / 2
        self.K = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])
        self._last = max(track.poses)
        self._cache = {}

    def pose(self, f):
        p = self.track.poses.get(f) or self.track.poses[min(f, self._last)]
        fwd, right, up = ue_axes(*p.rot)
        return p.loc.astype(float), np.stack([right, -up, fwd])

    def P(self, f):
        if f not in self._cache:
            C, Rm = self.pose(f)
            self._cache[f] = self.K @ np.hstack([Rm, (-Rm @ C)[:, None]])
        return self._cache[f]
