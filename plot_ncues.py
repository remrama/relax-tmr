"""Plot the amount of cues occuring in SWS for each participant."""
from pathlib import Path

from bids import BIDSLayout
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import utils


utils.set_matplotlib_style("technical")


root_dir = Path(utils.config["bids_root"])

cue_order = utils.config["cue_order"]
cue_palette = utils.config["cue_palette"]
participant_palette = utils.load_participant_palette()

y_column = "n_sws_cues"
s_column = "n_cues"
figsize = (2, 3)
jitter = 0.1
np.random.seed(2)

import_path = root_dir / "derivatives" / "task-sleep_ncues.tsv"
export_path = import_path.with_suffix(".png")

df = pd.read_csv(import_path, index_col="participant_id", sep="\t")
pp = utils.load_participants_file()
df = df.join(pp["tmr_condition"])
# desc = (df
#     .groupby(["participant_id", "description"])["n_cues"]
#     .agg(["count", "max", "mean", "std"])
# )
df["xval"] = df["tmr_condition"].map(cue_order.index)
df["xval"] += np.random.uniform(-jitter, jitter, size=len(df))
df["color"] = df.index.map(participant_palette)


desc = df.groupby("tmr_condition")[y_column].agg(["mean", "sem"])
desc["xval"] = desc.index.map(cue_order.index)
desc["color"] = desc.index.map(cue_palette)

fig, ax = plt.subplots(figsize=figsize)

bar_kwargs = {
    "width": 0.8,
    "edgecolor": "black",
    "linewidth": 1,
    "zorder": 1,
    "error_kw": dict(capsize=3, capthick=1, ecolor="black", elinewidth=1, zorder=2),
}
scatter_kwargs = {
    # "s": 30,
    "linewidths": 0.5,
    "edgecolors": "white",
    "clip_on": False,
    "zorder": 4,
    "alpha": 0.8
}

bars = ax.bar(data=desc, x="xval", height="mean", yerr="sem", color="color", **bar_kwargs)
bars.errorbar.lines[2][0].set_capstyle("round")
ax.scatter(data=df, x="xval", y=y_column, s=s_column, color="color", **scatter_kwargs)

# ax.set_ylabel("Probability of SWS during cue")
ax.set_ylabel(y_column)
ax.set_xlabel("TMR condition")
ax.margins(x=0.2)
# ax.set_ybound(upper=1)
ax.set_xticks(range(len(cue_order)))
ax.set_xticklabels(cue_order)
ax.tick_params(top=False, right=False, bottom=False)
ax.spines[["top", "right"]].set_visible(False)


# Export.
utils.export_mpl(export_path)
