"""Plot PVT."""
from pathlib import Path

from bids import BIDSLayout
import numpy as np
import pandas as pd
import pingouin as pg
import colorcet as cc

import matplotlib.pyplot as plt
import seaborn as sns

import utils


utils.set_matplotlib_style()

cue_order = utils.config["cue_order"]
cue_palette = utils.config["cue_palette"]
participant_palette = utils.load_participant_palette()
cue_labels = utils.config["cue_labels"]

jitter = 0.1
np.random.seed(1)
figsize = (2, 3)

bar_kwargs = {
    "width": 0.8,
    "edgecolor": "black",
    "linewidth": 1,
    "zorder": 1,
    "error_kw": dict(capsize=3, capthick=1, ecolor="black", elinewidth=1, zorder=2),
}
scatter_kwargs = {
    "s": 30,
    "linewidths": 0.5,
    "edgecolors": "white",
    "clip_on": False,
    "zorder": 4,
}

root_dir = Path(utils.config["bids_root"])
import_path = root_dir / "derivatives" / "task-pvt.tsv"
export_path = import_path.with_suffix(".png")

pp = utils.load_participants_file()
df = pd.read_csv(import_path, index_col="participant_id", sep="\t")
df = df.join(pp["tmr_condition"])

desc = df.groupby("tmr_condition")["mean"].agg(["mean", "sem"])

desc["xval"] = desc.index.map(cue_order.index)
desc["color"] = desc.index.map(cue_palette)

df["xval"] = df["tmr_condition"].map(cue_order.index)
df["xval"] += np.random.uniform(-jitter, jitter, size=len(df))
df["color"] = df.index.map(participant_palette)


fig, ax = plt.subplots(figsize=figsize)

# ax.axhline(0, color="black", linewidth=1, linestyle="dashed")

bars = ax.bar(data=desc, x="xval", height="mean", yerr="sem", color="color", **bar_kwargs)
bars.errorbar.lines[2][0].set_capstyle("round")

# Draw individual participants.
ax.scatter(data=df, x="xval", y="mean", c="color", **scatter_kwargs)


a, b = df.groupby("tmr_condition")["mean"].apply(list)
d = abs(pg.compute_effsize(a, b, paired=False, eftype="cohen"))
color = "black"
yline = 0.9
ax.hlines(
    y=yline,
    xmin=desc.at[cue_order[0], "xval"],
    xmax=desc.at[cue_order[1], "xval"],
    linewidth=0.5,
    color=color,
    capstyle="round",
    transform=ax.get_xaxis_transform(),
    clip_on=False,
)
# if p < 0.05:
#     ptext = "*" * sum([ p<cutoff for cutoff in (0.05, 0.01, 0.001) ])
# else:
#     ptext = fr"$p={p:.2f}$".replace("0", "", 1)
text = fr"$d={d:.02f}$"
ax.text(0.5, yline, text,
    color=color,
    transform=ax.transAxes,
    ha="center",
    va="bottom",
)

# Aesthetics
ax.set_xticks(desc.loc[cue_order, "xval"].to_numpy())
ax.set_xticklabels([cue_labels[c] for c in cue_order])
ax.margins(x=0.2, y=0.4)
ax.tick_params(top=False, bottom=False)
ax.set_ylabel("PVT reaction time (ms)")
# ax.set_ylabel(r"Post-sleep change in PVT reaction time ($\Delta$ ms)")
ax.set_xlabel("TMR cue sounds")

# Export.
utils.export_mpl(export_path)
