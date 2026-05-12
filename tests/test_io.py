import numpy as np
import pytest
import tifffile

from mrsnappy.io import read_volume, split_images, split_pairs


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
