"""SW params group"""
import argparse
from pathlib import Path

from bids import BIDSLayout
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns

import utils


utils.set_matplotlib_style()

metrics = ["Duration", "PTP", "Slope", "Frequency"]

ylimits = {
    "Duration": (1.05, 1.3),
    "PTP": (95, 135),
    "Slope": (380, 500),
    "Frequency": (0.84, 1),
}
ylabels = {
    "Duration": "Slow-wave Duration",
    "PTP": "Slow-wave Amplitude",
    "Slope": "Slow-wave Slope",
    "Frequency": "Slow-wave Frequency",
}

# Parse command-line arguments.
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--metric", type=str, default="Duration", choices=metrics)
parser.add_argument("-c", "--channel", type=str, default="Fz")
args = parser.parse_args()

metric = args.metric
channel = args.channel

root_dir = Path(utils.config["bids_root"])
layout = BIDSLayout(root_dir, derivatives=True, validate=False)
bids_files = layout.get(task="sleep", suffix="waves", extension="tsv")

export_path = root_dir / "derivatives" / f"task-sleep_waves_{metric}_{channel}.png"

df = pd.concat(
    [bf.get_df().assign(participant_id="sub-" + str(bf.entities["subject"])) for bf in bids_files],
    ignore_index=True,
)
df = df.set_index("participant_id")
pp = utils.load_participants_file()
df = df.join(pp["tmr_condition"])


pp_avgs = (df
    .groupby(["participant_id", "tmr_condition", "Channel"])[metrics].mean()
)
cue_avgs = (pp_avgs
    .groupby(["tmr_condition", "Channel"]).agg(["mean", "sem"])
    .stack(0).rename_axis(["tmr_condition", "channel", "metric"])
    .reset_index()
)

cue_order = utils.config["cue_order"]
cue_palette = utils.config["cue_palette"]
cue_labels = utils.config["cue_labels"]
participant_palette = utils.load_participant_palette()

cue_avgs["xval"] = cue_avgs["tmr_condition"].map(cue_order.index)
cue_avgs["color"] = cue_avgs["tmr_condition"].map(cue_palette)

jitter = 0.1
np.random.seed(1)

scatter_kwargs = {
    "s": 30,
    "linewidths": 0.5,
    "edgecolors": "white",
    "clip_on": False,
    "zorder": 4,
}
errorbar_kwargs = {
    "capsize": 3,
    "capthick": 1,
    "elinewidth": 1,
    "zorder": 2,
    "marker": "s",
    # "mec": "black",
    "ms": 8,
    # "mew": 4,
}


errorbar_data = (cue_avgs
    .query(f"channel=='{channel}'")
    .query(f"metric=='{metric}'")
    .set_index("tmr_condition")
)

pp_avgs = pp_avgs.reset_index()
pp_avgs["xval"] = pp_avgs["tmr_condition"].map(cue_order.index)
pp_avgs["xval"] += np.random.uniform(-jitter, jitter, size=len(pp_avgs))
pp_avgs["color"] = pp_avgs["participant_id"].map(participant_palette)

scatter_data = (pp_avgs
    .query(f"Channel=='{channel}'")
)

figsize = (2, 3)

fig, ax = plt.subplots(figsize=figsize)
for cue in cue_order:
    errorbars = ax.errorbar(
        data=errorbar_data.loc[cue],
        x="xval",
        y="mean",
        yerr="sem",
        color=cue_palette[cue],
        ecolor=cue_palette[cue],
        **errorbar_kwargs,
    )
    errorbars.lines[2][0].set_capstyle("round")

ax.scatter(data=scatter_data, x="xval", y=metric, c="color", **scatter_kwargs)


a, b = scatter_data.groupby("tmr_condition")[metric].apply(list)
d = abs(pg.compute_effsize(a, b, paired=False, eftype="cohen"))
color = "black"
ybar = 0.9
ax.hlines(
    y=ybar,
    xmin=errorbar_data.at[cue_order[0], "xval"],
    xmax=errorbar_data.at[cue_order[1], "xval"],
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
text = fr"$d={d:.2f}$".replace("0", "", 1)
ax.text(0.5, ybar, text,
    color=color,
    transform=ax.transAxes,
    ha="center",
    va="bottom",
)

ax.set_xlabel("TMR condition")
ax.set_xticks(range(len(cue_order)))
ax.set_xticklabels([cue_labels[c] for c in cue_order])
ax.set_ylabel(ylabels[metric])
ax.tick_params(bottom=False, top=False)
ax.margins(x=0.4)
ax.set_ylim(*ylimits[metric])


utils.export_mpl(export_path)
