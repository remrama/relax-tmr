"""Plot the amount of cues occuring in SWS for each participant."""
from pathlib import Path

from bids import BIDSLayout
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import utils


utils.set_matplotlib_style("technical")


################################################################################
# SETUP
################################################################################

root_dir = Path(utils.config["bids_root"])

layout = BIDSLayout(root_dir, derivatives=True, validate=False)
hypno_bids_files = layout.get(task="sleep", suffix="hypno", extension="tsv")
events_bids_files = layout.get(task="sleep", suffix="events", extension="tsv")
assert len(hypno_bids_files) == len(events_bids_files)

export_path = root_dir / "derivatives" / "cuedstages.png"

participants = utils.load_participants_file()
participant_palette = utils.load_participant_palette()
participant_conditions = utils.load_participants_file()["tmr_condition"].to_dict()

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

desc = (df
    .groupby(["participant_id", "description"])["n_cues"]
    .agg(["count", "max", "mean", "std"])
)

proba_columns = [c for c in df if c.startswith("proba")]
cued_epochs = df.query("n_cues > 0")
cued_desc = cued_epochs.groupby("participant_id")[proba_columns].agg(["count", "mean", "std"])
cued_desc = cued_desc.stack(0).reset_index(1).rename(columns={"level_1": "stage"})
cued_desc["stage"] = cued_desc["stage"].str.split("_").str[-1]
cued_desc = cued_desc.set_index("stage", append=True)

# melt = df.melt(
#     value_vars=proba_columns,
#     value_name="probability",
#     var_name="stage",
#     id_vars=["participant_id", "n_cues"],
# )
# melt["stage"] = melt["stage"].str.split("_").str[-1]
# cued_epochs = melt.query("n_cues > 0")
# cued_desc = cued_epochs.groupby(["participant_id", "stage"])["probability"].agg(["count", "mean", "std"])

n3 = cued_desc.loc[(slice(None), "N2"), :].droplevel("stage")
n3 = n3.join(participants["tmr_condition"])

x_order = ["relax", "story"]

jitter = 0.1
np.random.seed(2)

n3["xval"] = n3["tmr_condition"].map(x_order.index)
n3["xval"] += np.random.uniform(-jitter, jitter, size=len(n3))
n3["color"] = n3.index.map(participant_palette)

avgs = n3.groupby("tmr_condition")["mean"].agg(["mean", "sem"])
avgs["xval"] = avgs.index.map(x_order.index)

fig, ax = plt.subplots(figsize=(2, 3))

bars = ax.bar(data=avgs, x="xval", height="mean", yerr="sem", color="white", linewidth=0.5, edgecolor="black")
bars.errorbar.lines[2][0].set_capstyle("round")
ax.scatter(data=n3, x="xval", y="mean", s="count", color="color", linewidth=0.5, edgecolor="black", alpha=0.8, zorder=5)

ax.set_ylabel("Probability of SWS during cue")
ax.set_xlabel("TMR condition")
ax.margins(x=0.2)
ax.set_ybound(upper=0.44)
ax.set_xticks(range(len(x_order)))
ax.set_xticklabels(x_order)
ax.tick_params(top=False, right=False, bottom=False)
ax.spines[["top", "right"]].set_visible(False)


# Export.
utils.export_mpl(export_path)
