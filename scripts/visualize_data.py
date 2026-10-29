#!/usr/bin/env python3
"""Export sonar previews and sensor trajectories without CUDA or a display.

Only load trusted datasets: the original data format uses Python pickle.
"""
import argparse
import json
import pickle
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def natural_key(path):
    return [int(part) if part.isdigit() else part
            for part in re.split(r"(\d+)", path.name)]


def visualize(dataset, output, samples):
    with (dataset / "Config.json").open() as handle:
        config = json.load(handle)
    sonar = next(sensor["configuration"]
                 for agent in config["agents"] for sensor in agent["sensors"]
                 if sensor["sensor_type"] == "ImagingSonar")
    files = sorted((dataset / "Data").glob("*.pkl"), key=natural_key)
    if not files:
        raise ValueError("No .pkl frames found in {}".format(dataset / "Data"))
    indices = np.unique(np.linspace(0, len(files) - 1,
                                   min(samples, len(files)), dtype=int))
    images, poses = {}, []
    for index, path in enumerate(files):
        with path.open("rb") as handle:
            state = pickle.load(handle)
        pose = np.asarray(state["PoseSensor"])
        if pose.shape != (4, 4):
            raise ValueError("Invalid PoseSensor shape in {}".format(path))
        poses.append(pose[:3, 3].copy())
        if index in indices:
            image = np.asarray(state["ImagingSonar"])
            if image.ndim != 2:
                raise ValueError("Invalid sonar image shape in {}".format(path))
            images[index] = image.copy()

    destination = output / dataset.name
    destination.mkdir(parents=True, exist_ok=True)
    # Shared intensity limits make raw/processed frames directly comparable.
    vmin = min(float(image.min()) for image in images.values())
    vmax = max(float(image.max()) for image in images.values())
    fig, axes = plt.subplots(2, len(indices), figsize=(3.5 * len(indices), 8),
                             squeeze=False, constrained_layout=True)
    extent = [-sonar["Azimuth"] / 2, sonar["Azimuth"] / 2,
              sonar["RangeMin"], sonar["RangeMax"]]
    for column, index in enumerate(indices):
        raw = images[index]
        processed = raw.copy()
        # Match load_data.py exactly; do not alter the source data.
        processed[processed < 0.2] = 0
        processed[processed.shape[0] - 200:, :] = 0
        for row, (label, image) in enumerate((("Raw", raw),
                                               ("Training preprocessing", processed))):
            ax = axes[row, column]
            artist = ax.imshow(image, origin="lower", extent=extent,
                               aspect="auto", cmap="magma", vmin=vmin, vmax=vmax)
            ax.set_title("{}\n{}".format(files[index].name, label))
            ax.set_xlabel("Azimuth (degrees)")
            ax.set_ylabel("Range (m)")
    fig.colorbar(artist, ax=axes.ravel().tolist(), label="Sonar intensity", shrink=0.8)
    fig.suptitle("{} | {} frames | Elevation FOV: {} degrees".format(
        dataset.name, len(files), sonar["Elevation"]))
    fig.savefig(str(destination / "sonar_samples.png"), dpi=150)
    plt.close(fig)

    positions = np.asarray(poses)
    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    ax3 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122)
    ax3.plot(*positions.T, color="gray", linewidth=0.6, alpha=0.5)
    points = ax3.scatter(*positions.T, c=np.arange(len(files)), s=5, cmap="viridis")
    ax3.set(xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)", title="Sensor positions (3D)")
    ax3.set_box_aspect(np.maximum(np.ptp(positions, axis=0), 0.1))
    ax2.plot(positions[:, 0], positions[:, 1], color="gray", linewidth=0.6)
    ax2.scatter(positions[:, 0], positions[:, 1], c=np.arange(len(files)), s=5, cmap="viridis")
    ax2.scatter(*positions[0, :2], marker="*", color="red", s=100, label="First frame")
    ax2.set(xlabel="X (m)", ylabel="Y (m)", title="Sensor positions (XY)")
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.legend()
    fig.colorbar(points, ax=[ax3, ax2], label="Frame index (natural filename order)", shrink=0.8)
    fig.suptitle(dataset.name)
    fig.savefig(str(destination / "trajectory.png"), dpi=150)
    plt.close(fig)
    summary = dict(dataset=dataset.name, frames=len(files),
                   image_shape=list(images[indices[0]].shape), sonar=sonar,
                   sampled_files=[files[i].name for i in indices],
                   position_min=positions.min(axis=0).tolist(),
                   position_max=positions.max(axis=0).tolist())
    with (destination / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print("{}: {} frames -> {}".format(dataset.name, len(files), destination), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/data/jl/data/neusis"))
    parser.add_argument("--dataset", default="all", help="Dataset directory name, or all")
    parser.add_argument("--samples", type=int, default=4, help="Evenly spaced preview frames (1-12)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "outputs" / "data_preview")
    args = parser.parse_args()
    if not 1 <= args.samples <= 12:
        parser.error("--samples must be between 1 and 12")
    if not args.data_root.is_dir():
        parser.error("Data root does not exist: {}".format(args.data_root))
    datasets = (sorted(p.parent for p in args.data_root.glob("*/Config.json"))
                if args.dataset == "all" else [args.data_root / args.dataset])
    if not datasets:
        parser.error("No datasets containing Config.json found")
    for dataset in datasets:
        if not (dataset / "Config.json").is_file():
            parser.error("Missing Config.json in {}".format(dataset))
        visualize(dataset, args.output_dir, args.samples)


if __name__ == "__main__":
    main()
