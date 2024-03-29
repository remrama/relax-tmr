"""Plot subjective response from debriefing questionnaire in the morning."""
import argparse
from pathlib import Path

from bids import BIDSLayout
import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns

import utils


utils.set_matplotlib_style()

cue_order = utils.config["cue_order"]
cue_palette = utils.config["cue_palette"]
cue_labels = utils.config["cue_labels"]
participant_palette = utils.load_participant_palette()


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--column", type=str, default="Alertness_1")
args = parser.parse_args()

column = args.column

root_dir = Path(utils.config["bids_root"])
import_path = root_dir / "phenotype" / "debriefing.tsv"
export_path = root_dir / "derivatives" / f"response-{column}.png"

# layout = BIDSLayout(root_dir, validate=False)
# bids_file = layout.get(suffix="debriefing", extension="tsv")[0]
# df = bids_file.get_df()
df = pd.read_csv(import_path, index_col="participant_id", sep="\t")
meta = utils.import_json(import_path.with_suffix(".json"))

participants = utils.load_participants_file()
df = df.join(participants)

for survey in ["PANAS", "STAI", "TAS"]:
    df = utils.agg_questionnaire_columns(df, survey, delete_cols=True)

desc = (df
    .groupby("tmr_condition")[column]
    .agg(["count", "mean", "std", "sem", "min", "median", "max"])
)


desc = desc.loc[cue_order]

yvals = desc["mean"].to_numpy()
yerrs = desc["sem"].to_numpy()
xvals = np.arange(yvals.size)
color = desc.index.map(cue_palette)


figsize = (1.7, 2.5)

lines_kwargs = dict(linewidths=0.5, zorder=3)
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

fig, ax = plt.subplots(figsize=figsize)
bars = ax.bar(xvals, yvals, yerr=yerrs, color=color, **bar_kwargs)
bars.errorbar.lines[2][0].set_capstyle("round")

# Draw individual participants.
jitter = 0.1
np.random.seed(1)
participant_palette = utils.load_participant_palette()
df["xval"] = df["tmr_condition"].map(lambda x: cue_order.index(x))
df["xval"] += np.random.uniform(-jitter, jitter, size=len(df))
df["color"] = df.index.map(participant_palette)#.to_numpy()
ax.scatter("xval", column, c="color", data=df, **scatter_kwargs)

a, b = df.groupby("tmr_condition")[column].apply(list)
ttest = pg.ttest(a, b, paired=False)
mwu = pg.mwu(a, b)
d = abs(ttest.at["T-test", "cohen-d"])
p = ttest.at["T-test", "p-val"]
# pcolor ="black" if p < 0.1 else "gainsboro"
ybar = 0.9
utils.draw_significance_bar(
    ax=ax,
    x1=0,
    x2=1,
    y=ybar,
    p=p,
    height=0.02,
    caplength=None,
    linewidth=1,
)
# if p < 0.05:
#     ptext = "*" * sum([ p<cutoff for cutoff in (0.05, 0.01, 0.001) ])
# else:
#     ptext = fr"$p={p:.2f}$".replace("0", "", 1)
color = "black" if p < 0.05 else "gainsboro"
text = fr"$d={d:.02f}$"
ax.text(0.5, ybar + 0.025, text,
    color=color,
    transform=ax.transAxes,
    ha="center",
    va="bottom",
)

# Aesthetics
if column.startswith("Alertness"):
    ax.set_ybound(upper=100)
try:
    ylabel = meta[column]["Probe"]
except:
    ylabel = column
if column == "Alertness_1":
    ylabel = "How alert are you this morning?"
ax.set_ylabel(ylabel)
ax.set_xlabel("TMR Cues")
ax.tick_params(top=False, bottom=False, right=False)
ax.grid(False)
ax.set_xticks(xvals)
ax.set_xticklabels([cue_labels[c] for c in cue_order])
ax.margins(x=0.2)
ax.spines[["top", "right"]].set_visible(False)
# ax.spines[["left","bottom"]].set_position(("outward", 5))


# Export.
utils.export_mpl(export_path)
