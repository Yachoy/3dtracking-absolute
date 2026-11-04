"""Парсер CSV камер: формат Way1 и формат Way2 (фокус построчно, сенсор в matrix_size.txt)."""
from core.camera import Camera
from core.dataset import CameraTrack, read_transform_csv

WAY1 = ("Frame;Loc_X;Loc_Y;Loc_Z;Rot_Roll;Rot_Pitch;Rot_Yaw;;;Focal length;35\n"
        "0;-24701.9592;6623.5717;1200.0;0.0;0.0;-70.0;;;матрица;23,76*13,365мм\n"
        "1;-24701.9592;6623.5717;1200.0;0.0;0.0;-70.0;;;;\n")
WAY2 = ("Frame;Loc_X;Loc_Y;Loc_Z;Rot_Roll;Rot_Pitch;Rot_Yaw;Focal_Length_mm\n"
        "0;-36539.0;-23754.0762;25898.4727;0.0;-49.0;33.0;60.0\n"
        "1;-36530.0;-23750.0;25898.4727;0.0;-49.0;33.0;70.0\n")


def test_way1_header_focal_and_inline_sensor(tmp_path):
    path = tmp_path / "object_transform_CineCameraActor1.csv"
    path.write_bytes(WAY1.encode("cp1251"))
    poses, meta = read_transform_csv(path)
    assert meta["focal_mm"] == 35.0 and meta["sensor_mm"] == (23.76, 13.365)
    assert not meta["focal_per_frame"] and poses[1].focal_mm is None


def test_way2_per_frame_focal_and_sensor_file(tmp_path):
    path = tmp_path / "object_transform_CineCameraActor2.csv"
    path.write_bytes(WAY2.encode("cp1251"))
    (tmp_path / "matrix_size.txt").write_text("23,76 mm * 13,365 mm", encoding="utf-8")
    poses, meta = read_transform_csv(path)
    assert meta["focal_mm"] == 60.0 and meta["sensor_mm"] == (23.76, 13.365) and meta["focal_per_frame"]
    assert [poses[f].focal_mm for f in (0, 1)] == [60.0, 70.0]
    track = CameraTrack(2, meta["focal_mm"], meta["sensor_mm"], poses)
    # фокус кадра, а не первый в файле: зум меняет fx
    assert Camera.from_track(track, 1, 1920, 1080).fx == 1920 * 70.0 / 23.76


def test_way2_without_sensor_file_is_incomplete(tmp_path):
    path = tmp_path / "object_transform_CineCameraActor3.csv"
    path.write_bytes(WAY2.encode("cp1251"))
    assert read_transform_csv(path)[1]["sensor_mm"] is None
