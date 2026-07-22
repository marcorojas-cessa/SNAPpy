import numpy as np
import pytest
import tifffile

from mrsnappy.io import read_points_csv, read_volume, split_images, split_pairs, write_points_csv


def test_read_volume_rejects_nan_or_inf_voxels(tmp_path) -> None:
    image_path = tmp_path / "bad_image.tif"
    volume = np.zeros((3, 3, 3), dtype=np.float32)
    volume[1, 1, 1] = np.nan
    tifffile.imwrite(image_path, volume)

    with pytest.raises(ValueError, match="finite-valued images"):
        read_volume(image_path)


def test_dataset_split_helpers_accept_tif_and_tiff(tmp_path) -> None:
    split_root = tmp_path / "train"
    split_root.mkdir()
    for stem, suffix in (("image_a", ".tif"), ("image_b", ".tiff")):
        tifffile.imwrite(split_root / f"{stem}{suffix}", np.zeros((3, 3, 3), dtype=np.float32))
        (split_root / f"{stem}.csv").write_text("x,y,z\n1,1,1\n")

    assert [path.name for path in split_images(tmp_path, "train")] == ["image_a.tif", "image_b.tiff"]
    assert [image.name for image, _ in split_pairs(tmp_path, "train")] == ["image_a.tif", "image_b.tiff"]


def test_read_points_csv_accepts_native_2d_columns(tmp_path) -> None:
    xy_path = tmp_path / "xy.csv"
    xy_path.write_text("x,y\n4,3\n8,7\n")

    axis_path = tmp_path / "axis.csv"
    axis_path.write_text("axis-0,axis-1\n11,12\n")

    assert np.array_equal(read_points_csv(xy_path), np.asarray([[3, 4], [7, 8]], dtype=np.float32))
    assert np.array_equal(read_points_csv(axis_path, ndim=2), np.asarray([[11, 12]], dtype=np.float32))


def test_write_points_csv_uses_2d_or_3d_coordinate_columns(tmp_path) -> None:
    path_2d = tmp_path / "points_2d.csv"
    path_3d = tmp_path / "points_3d.csv"

    write_points_csv(path_2d, np.asarray([[3, 4]], dtype=np.float32), np.asarray([0.5], dtype=np.float32))
    write_points_csv(path_3d, np.asarray([[2, 3, 4]], dtype=np.float32), np.asarray([0.7], dtype=np.float32))

    assert path_2d.read_text().splitlines()[0] == "x,y,score"
    assert path_3d.read_text().splitlines()[0] == "x,y,z,score"
