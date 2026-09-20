"""Small deterministic fixtures for smoke tests; never scientific evidence."""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def create_synthetic(root):
    root = Path(root)
    data, masks = root / "images", root / "segmentations"
    data.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(91)
    rows = []
    for split, count in ((0, 4), (1, 2), (2, 2)):
        for label in (0, 1):
            for place in (0, 1):
                for i in range(count):
                    filename = f"split{split}_y{label}_place{place}_{i}.jpg"
                    image = np.zeros((256, 256, 3), dtype=np.uint8)
                    image[:] = [40, 100, 35] if place == 0 else [40, 60, 140]
                    mask = np.zeros((256, 256), dtype=np.uint8)
                    # Large enough to exercise Rescue-20, including partial patch edges.
                    mask[48:208, 64:192] = 255
                    image[mask > 0] = [190, 90, 65] if label == 0 else [230, 230, 210]
                    noise = rng.integers(0, 10, image.shape, dtype=np.uint8)
                    Image.fromarray(image + noise).save(data / filename)
                    Image.fromarray(mask).save(masks / Path(filename).with_suffix(".png"))
                    rows.append({"img_filename": filename, "y": label, "place": place, "split": split})
    pd.DataFrame(rows).to_csv(data / "metadata.csv", index=False)
    return str(data), str(masks)
