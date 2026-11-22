from pathlib import Path

import numpy as np
import torch
import yaml
from astropy.io import fits

import torchmfbd

project = Path("/srv/zltong721/projects/torchmfbd")
input_path = project / "data/inputs/NGC_6167_GaiaDR3.fits"
template_path = project / "data/outputs/config.yaml"

# 只从主 HDU 读取输入序列。
with fits.open(input_path, memmap=False) as hdul:
    header = hdul[0].header.copy()
    cube = np.array(hdul[0].data, dtype=np.float32, copy=True)

if cube.shape != (64, 256, 256) or not np.isfinite(cube).all():
    raise ValueError(f"输入帧尺寸或数值异常，实际尺寸为 {cube.shape}")

# (frame, y, x) -> (sequence, frame, y, x)
frames = torch.from_numpy(cube[None, ...])

with template_path.open(encoding="utf-8") as file:
    config = yaml.safe_load(file)

diameter_cm = float(header["DIAM_M"]) * 100.0
config["telescope"]["diameter"] = diameter_cm
config["telescope"]["central_obscuration"] = 0.30 * diameter_cm
config["images"]["n_pixel"] = cube.shape[-1]
config["images"]["pix_size"] = float(header["PIXSCALE"])
config["object1"]["wavelength"] = float(header["WAVE_NM"]) * 10.0
config["optimization"]["gpu"] = -1  # 使用 CPU，有可用的 CUDA GPU 时可改成 0

dec = torchmfbd.Deconvolution(config)
dec.add_frames(frames, id_object=0, id_diversity=0, diversity=0.0)
dec.deconvolve(
    infer_object=False,
    optimizer="adam",
    simultaneous_sequences=1,
    n_iterations=100,
)

reconstruction = dec.obj[0][0].detach().cpu().numpy()
output = Path("NGC_6167_reconstruction.fits")
fits.writeto(output, reconstruction)
print(f"重建尺寸 {reconstruction.shape}，输出 {output.resolve()}")
