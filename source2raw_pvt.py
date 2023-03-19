"""Convert raw matlab PVT files to BIDS tsvs."""
import argparse
from pathlib import Path

from bids import BIDSLayout
import numpy as np
import pandas as pd
import mat73

import utils


parser = argparse.ArgumentParser()
parser.add_argument(
    "-p", "--participant", type=int, required=True, choices=utils.participant_values()
)
args = parser.parse_args()


################################################################################
# SETUP
################################################################################


participant = args.participant
participant_id = f"sub-{participant:03d}"

root_dir = Path(utils.config["bids_root"])
import_paths = sorted((root_dir / "sourcedata" / participant_id).glob("*/trial.mat"))
assert len(import_paths) == 2

for acq, path in zip(["pre", "post"], import_paths):
    pvt = mat73.loadmat(path)
    stats = pvt["stats"]
    data = pvt["data"]
    conf = pvt["conf"]
    conf = {k: v for k, v in conf.items() if k.endswith("mood")}
    # Mood probe: How sleepy are you? (1 = not at all, 10 = very)

    df_pvt = pd.DataFrame(data)

    df_sleepy = (pd.Series(conf)
        .rename("sleepiness")
        .rename_axis("timing")
        .reset_index()
        .replace({"timing": {"pre_mood": "before_pvt", "post_mood": "after_pvt"}})
    )

    # Export.
    export_path_pvt = root_dir / participant_id / "beh" / f"{participant_id}_task-pvt_acq-{acq}_beh.tsv"
    export_path_sleepy = root_dir / participant_id / "beh" / f"{participant_id}_task-sleepy_acq-{acq}_beh.tsv"
    utils.export_tsv(df_pvt, export_path_pvt, index=False)
    utils.export_tsv(df_sleepy, export_path_sleepy, index=False)
