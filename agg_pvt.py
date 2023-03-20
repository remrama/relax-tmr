"""Analyze PVT, pre/post analyze and plot."""
from pathlib import Path

from bids import BIDSLayout
import pandas as pd

import utils


root_dir = Path(utils.config["bids_root"])
export_path = root_dir / "derivatives" / "task-pvt.tsv"
layout = BIDSLayout(root_dir, derivatives=False, validate=False)
bids_files = layout.get(task="pvt", suffix="beh", extension="tsv")

dataframe_list = []
for bf in bids_files:
    participant_id = "sub-" + bf.entities["subject"]
    acquisition_id = "acq-" + bf.entities["acquisition"]
    data = bf.get_df().assign(participant_id=participant_id, acquisition_id=acquisition_id)
    dataframe_list.append(data)
# df = pd.concat(
#     [bf.get_df().assign(participant_id="sub-" + str(bf.entities["subject"])) for bf in bids_files],
#     ignore_index=True,
# )

df = pd.concat(dataframe_list, ignore_index=True).set_index(["participant_id", "acquisition_id"])

### I think "bad" column combines "bp" and "fs" columns.
# Ignore false starts (only a few), nr (no response?), and bp (bad press?) for now.
# Remove bad trials so they don't influence mean.
df = df.query("bad.eq(False)").drop(columns=["bad", "bp", "fs", "nr"])
desc = df.groupby(level=["participant_id", "acquisition_id"])["rt"].describe()
    # .agg(["count", "mean", "std", "sem", "min", "median", "max"])



pre = desc.loc[(slice(None), "acq-pre", slice(None)), :].droplevel("acquisition_id")
post = desc.loc[(slice(None), "acq-post", slice(None)), :].droplevel("acquisition_id")
diff = post - pre

utils.export_tsv(diff, export_path)
