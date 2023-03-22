"""Plot a single sleep statistic."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import utils


utils.set_matplotlib_style()

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--sstat", type=str, default="SME")
args = parser.parse_args()

sstat = args.sstat

cue_order = utils.config["cue_order"]
cue_palette = utils.config["cue_palette"]
cue_labels = utils.config["cue_labels"]
participant_palette = utils.load_participant_palette()

figsize = (2.5, 3)
jitter = 0.1
np.random.seed(2)

root_dir = Path(utils.config["bids_root"])
import_path = root_dir / "derivatives" / f"task-sleep_sstats.tsv"
export_path = root_dir / "derivatives" / f"task-sleep_sstats_{sstat}.png"


df = pd.read_csv(import_path, index_col="participant_id", sep="\t")
pp = utils.load_participants_file()
df = df.join(pp["tmr_condition"])

df["xval"] = df["tmr_condition"].map(cue_order.index)
df["xval"] += np.random.uniform(-jitter, jitter, size=len(df))
df["color"] = df.index.map(participant_palette)

desc = df.groupby("tmr_condition")[sstat].agg(["mean", "sem"])
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
    "s": 30,
    "linewidths": 0.5,
    "edgecolors": "white",
    "clip_on": False,
    "zorder": 4,
    "alpha": 0.8
}


bars = ax.bar(data=desc, x="xval", height="mean", yerr="sem", color="color", **bar_kwargs)
bars.errorbar.lines[2][0].set_capstyle("round")
ax.scatter(data=df, x="xval", y=sstat, color="color", **scatter_kwargs)

ax.set_xticks(range(len(cue_order)))
ax.set_xticklabels([cue_labels[x] for x in cue_order])
ax.margins(x=0.2)
# ax.set_ybound(upper=100)
ax.tick_params(top=False, bottom=False, right=False)
ax.set_ylabel(sstat)
ax.set_xlabel("TMR condition")

# Export.
utils.export_mpl(export_path)
