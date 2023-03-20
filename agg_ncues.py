"""Plot the amount of cues occuring in SWS for each participant."""
from pathlib import Path

from bids import BIDSLayout
import numpy as np
import pandas as pd

import utils


root_dir = Path(utils.config["bids_root"])

layout = BIDSLayout(root_dir, derivatives=True, validate=False)
hypno_bids_files = layout.get(task="sleep", suffix="hypno", extension="tsv")
events_bids_files = layout.get(task="sleep", suffix="events", extension="tsv")
assert len(hypno_bids_files) == len(events_bids_files)

export_path = root_dir / "derivatives" / "task-sleep_ncues.tsv"


dataframe_list = []
for h_bf, e_bf in zip(hypno_bids_files, events_bids_files):
    participant_id = "sub-" + h_bf.entities["subject"]
    hypno = h_bf.get_df()
    events = e_bf.get_df()

    hypno = hypno.set_index("onset")
    cue_onsets = events.query("description.str.endswith('.wav')")["onset"].to_numpy()
    idx = hypno.index.get_indexer(cue_onsets, method="ffill")
    counts = hypno.index[idx].value_counts().rename("n_cues")
    hypno = hypno.join(counts)
    hypno["n_cues"] = hypno["n_cues"].fillna(0).astype(int)
    # counts = hypno.iloc[idx]["description"].value_counts().rename_axis("stage").rename(participant_id)
    hypno = hypno.reset_index().assign(participant_id=participant_id)
    dataframe_list.append(hypno)

# df = pd.concat(series_list, axis=1).fillna(0).astype(int)
# df = df.T.rename_axis("participant_id").rename_axis(None, axis=1).sort_index(axis=1)
df = pd.concat(dataframe_list)
# desc = (df
#     .groupby(["participant_id", "description"])["n_cues"]
#     .agg(["count", "max", "mean", "std"])
# )

df["n_cuesXproba_N3"] = df["n_cues"].mul(df["proba_N3"])
df["cuedXproba_N3"] = df["n_cues"].gt(0).mul(df["proba_N3"])

# df = df.query("n_cues > 0")
desc = df.groupby("participant_id")[["n_cues", "n_cuesXproba_N3", "cuedXproba_N3"]].sum()
n_sws_cues = (df
    .query("description == 'N3'")
    .groupby("participant_id")["n_cues"]
    .agg(["count", "sum"])
    .rename(columns={"count": "n_sws_cued_epochs", "sum": "n_sws_cues"})
)
desc = desc.join(n_sws_cues)
desc[["n_sws_cued_epochs", "n_sws_cues"]] = desc[["n_sws_cued_epochs", "n_sws_cues"]].fillna(0).astype(int)


# Export.
utils.export_tsv(desc, export_path)
