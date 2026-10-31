#!/usr/bin/env python3
"""Plot trusted sonar pickle frames as range/azimuth images and polar fans."""
import argparse
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from visualize_data import natural_key


def plot_dataset(dataset, output, samples, frame):
    with (dataset / "Config.json").open() as handle:
        config = json.load(handle)
    sonar = next(s["configuration"] for a in config["agents"]
                 for s in a["sensors"] if s["sensor_type"] == "ImagingSonar")
    files = sorted((dataset / "Data").glob("*.pkl"), key=natural_key)
    if not files:
        raise ValueError("No frames in {}".format(dataset))
    if frame is not None:
        selected = [p for p in files if p.name == frame]
        if not selected:
            raise ValueError("Frame {} not found in {}".format(frame, dataset))
    else:
        indices = np.unique(np.linspace(0, len(files) - 1,
                                       min(samples, len(files)), dtype=int))
        selected = [files[i] for i in indices]
    images = []
    for path in selected:
        with path.open("rb") as handle:
            image = np.asarray(pickle.load(handle)["ImagingSonar"])
        if image.ndim != 2 or not np.isfinite(image).all():
            raise ValueError("Expected a finite 2D sonar image: {}".format(path))
        images.append(image)
    vmin = min(0.0, min(float(im.min()) for im in images))
    vmax = max(float(im.max()) for im in images)
    half_angle = float(sonar["Azimuth"]) / 2
    rmin, rmax = float(sonar["RangeMin"]), float(sonar["RangeMax"])
    if not (0 < half_angle < 90 and 0 <= rmin < rmax):
        raise ValueError("Expected forward-looking sonar with valid range/FOV")

    fig, axes = plt.subplots(len(images), 3, squeeze=False,
                             figsize=(11, 4.5 * len(images)), constrained_layout=True)
    for row, (path, raw) in enumerate(zip(selected, images)):
        processed = raw.copy()
        processed[processed < 0.2] = 0
        processed[processed.shape[0] - 200:, :] = 0
        # Pixel edges span the configured bins; no interpolation of intensities.
        angles = np.deg2rad(np.linspace(-half_angle, half_angle, raw.shape[1] + 1))
        ranges = np.linspace(rmin, rmax, raw.shape[0] + 1)
        theta, radius = np.meshgrid(angles, ranges)
        x, y = radius * np.sin(theta), radius * np.cos(theta)
        ax = axes[row, 0]
        ax.imshow(raw, origin="lower", aspect="auto", cmap="magma",
                  extent=[-half_angle, half_angle, rmin, rmax], vmin=vmin, vmax=vmax)
        ax.set(title="{}\nRaw range-azimuth".format(path.name),
               xlabel="Azimuth (degrees)", ylabel="Range (m)")
        for column, (label, im) in enumerate((("Raw polar fan", raw),
                                               ("Preprocessed polar fan", processed)), 1):
            ax = axes[row, column]
            artist = ax.pcolormesh(x, y, im, shading="flat", cmap="magma",
                                   vmin=vmin, vmax=vmax, rasterized=True)
            # Range arcs and radial guides are drawn in the sonar-local frame.
            for distance in np.linspace(rmin, rmax, 5)[1:]:
                ax.plot(distance * np.sin(angles), distance * np.cos(angles),
                        color="gray", alpha=0.45, linewidth=0.6)
            for angle in (-half_angle, 0, half_angle):
                a = np.deg2rad(angle)
                ax.plot([rmin * np.sin(a), rmax * np.sin(a)],
                        [rmin * np.cos(a), rmax * np.cos(a)],
                        color="gray", alpha=0.45, linewidth=0.6)
            ax.plot(0, 0, "c^", markersize=5)
            ax.set(title="{}\n{}".format(path.name, label),
                   xlabel="Lateral x (m)", ylabel="Forward y (m)",
                   xlim=(-rmax * np.sin(angles[-1]) * 1.08,
                         rmax * np.sin(angles[-1]) * 1.08),
                   ylim=(-0.02 * rmax, rmax * 1.03))
            ax.set_aspect("equal")
    fig.colorbar(artist, ax=axes.ravel().tolist(), label="Sonar intensity", shrink=0.7)
    fig.suptitle("{} | Horizontal FOV: {} deg | Elevation FOV: {} deg\n"
                 "Fan display assumes zero elevation; not a world-coordinate slice".format(
                     dataset.name, sonar["Azimuth"], sonar["Elevation"]))
    destination = output / dataset.name
    destination.mkdir(parents=True, exist_ok=True)
    name = "polar_{}.png".format(Path(frame).stem) if frame else "polar_samples.png"
    fig.savefig(str(destination / name), dpi=150)
    plt.close(fig)
    print("{} -> {}".format(", ".join(p.name for p in selected), destination / name))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/data/jl/data/neusis"))
    parser.add_argument("--dataset", default="14deg_submarine", help="Dataset name, or all")
    parser.add_argument("--samples", type=int, default=4, help="Evenly spaced frames (1-12)")
    parser.add_argument("--frame", help="Exact filename, e.g. 100.pkl; overrides --samples")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "outputs" / "polar_preview")
    args = parser.parse_args()
    if not 1 <= args.samples <= 12:
        parser.error("--samples must be between 1 and 12")
    datasets = (sorted(p.parent for p in args.data_root.glob("*/Config.json"))
                if args.dataset == "all" else [args.data_root / args.dataset])
    if not datasets:
        parser.error("No datasets found under {}".format(args.data_root))
    for dataset in datasets:
        if not (dataset / "Config.json").is_file():
            parser.error("Missing Config.json in {}".format(dataset))
        plot_dataset(dataset, args.output_dir, args.samples, args.frame)


if __name__ == "__main__":
    main()
