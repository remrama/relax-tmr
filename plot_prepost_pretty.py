"""Plot interaction between pre/post measure and TMR condition."""
import argparse
from pathlib import Path

from bids import BIDSLayout
import colorcet as cc
from matplotlib.collections import LineCollection
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
parser.add_argument("-c", "--column", type=str, default="PANAS_neg")
args = parser.parse_args()

column = args.column

root_dir = Path(utils.config["bids_root"])
import_path_pre = root_dir / "phenotype" / "initial.tsv"
import_path_post = root_dir / "phenotype" / "debriefing.tsv"
export_path = root_dir / "derivatives" / f"prepost-{column}_PRETTY.png"

# layout = BIDSLayout(root_dir, validate=False)
# bids_file = layout.get(suffix="debriefing", extension="tsv")[0]
# df = bids_file.get_df()
pre = pd.read_csv(import_path_pre, index_col="participant_id", sep="\t")
post = pd.read_csv(import_path_post, index_col="participant_id", sep="\t")
pre_meta = utils.import_json(import_path_pre.with_suffix(".json"))
post_meta = utils.import_json(import_path_post.with_suffix(".json"))

# scrn_path = root_dir / "phenotype" / "screening.tsv"
# scrn = pd.read_csv(scrn_path, index_col="participant_id", sep="\t")
# scrn_meta = utils.import_json(scrn_path.with_suffix(".json"))

participants = utils.load_participants_file()
pre = pre.join(participants["tmr_condition"]).set_index("tmr_condition", append=True)
post = post.join(participants["tmr_condition"]).set_index("tmr_condition", append=True)

for survey in ["PANAS", "STAI"]:
    pre = utils.agg_questionnaire_columns(pre, survey)
    post = utils.agg_questionnaire_columns(post, survey)

pre = pre.assign(time="pre").set_index("time", append=True)
post = post.assign(time="post").set_index("time", append=True)

df = pd.concat([pre, post]).dropna(axis=1)
df = df.reset_index()

anova = pg.mixed_anova(data=df, between="tmr_condition", dv=column, within="time", subject="participant_id")
pwise = pg.pairwise_tests(
    data=df,
    between="tmr_condition",
    dv=column,
    within="time",
    subject="participant_id",
    within_first=False,
    return_desc=False,
    parametric=False,
    effsize="cohen",
)

desc = df.groupby(["tmr_condition", "time"])[column].agg(["mean", "sem"]).reset_index()
desc["xval"] = desc["tmr_condition"].map(cue_order.index).multiply(2.5)
desc["xval"] = desc["xval"].add(desc["time"].eq("post").astype(int))
desc["color"] = desc["tmr_condition"].map(cue_palette)

bar_kwargs = {
    "width": 1,
    "edgecolor": "black",
    "linewidth": 1,
    "zorder": 1,
    "error_kw": dict(capsize=3, capthick=1, ecolor="black", elinewidth=1, zorder=2),
}

figsize = (2.5, 2)
fig, ax = plt.subplots(figsize=figsize)
bars = ax.bar(data=desc, x="xval", height="mean", yerr="sem", color="color", **bar_kwargs)
bars.errorbar.lines[2][0].set_capstyle("round")

jitter = 0.1
np.random.seed(1)
df["color"] = df["participant_id"].map(participant_palette)
df["xval"] = df["tmr_condition"].map(cue_order.index).multiply(2.5)
macro_xvals = df["xval"].sort_values().unique().tolist()
df["xval"] = df["xval"].add(df["time"].eq("post").astype(int))
micro_xvals = df["xval"].sort_values().unique().tolist()
df["xval"] = df["xval"].add(np.random.uniform(-jitter, jitter, size=len(df)))
scatter_kwargs = {
    "s": 50,
    "linewidths": 0.5,
    "edgecolors": "white",
    "clip_on": False,
    "zorder": 4,
}
ax.scatter(data=df, x="xval", y=column, c="color", **scatter_kwargs)


lines_kwargs = {"linewidths": 1, "zorder": 3}
table = (df
    .pivot(index=["participant_id", "tmr_condition"], columns="time", values=column)
    .rename_axis(None, axis=1).reset_index()
)
xtable = (df
    .pivot(index=["participant_id", "tmr_condition"], columns="time", values="xval")
    .rename_axis(None, axis=1).reset_index()
)
line_colors = table["participant_id"].map(participant_palette)
line_xvals = xtable[["pre", "post"]].to_numpy()
line_yvals = table[["pre", "post"]].to_numpy()
line_segments = [np.column_stack([xvals, yvals]) for xvals, yvals in zip(line_xvals, line_yvals)]
lines = LineCollection(
    line_segments,
    colors=line_colors,
    label=table.index.to_numpy(),
    # offsets=offsets, offset_transform=None,
    **lines_kwargs,
)
ax.add_collection(lines)
# scatterx, scattery = np.row_stack(segments).T
# scatterc = np.repeat(colors, 2)
# ax.scatter(scatterx, scattery, c=scatterc, **scatter_kwargs)

xticks = desc["xval"].sort_values().to_list()
# xticklabels = desc.sort_values("xval")["time"].to_list()
xticklabels = ["Before\nSleep", "After\nSleep"] * 2
ax.set_xticks(micro_xvals)
ax.set_xticklabels(xticklabels)

try:
    ylabel = meta[column]["Probe"]
except:
    ylabel = column
if column == "Alertness_1":
    ylabel = "How alert are you this morning?"
elif column == "SSS":
    ylabel = "Stanford Sleepiness Survey\n" + r"Awake $\leftarrow$     $\rightarrow$ Tired"
elif column == "PANAS_neg":
    ylabel = "PANAS Negative Affect"
ax.set_ylabel(ylabel)

if column in pre_meta:
    probe_lvls = pre_meta[column]["Levels"]
    yticks, yticklabels = zip(*[ (int(x[0]), y) for x, y in probe_lvls.items() ])
    yticklabels = [ y.replace("Very Unrelaxed", "Not Relaxed") for y in yticklabels ]
    yticks_major = [min(yticks), max(yticks)]
    yticklabels = [yticklabels[y-1] for y in yticks_major]
    if column == "SSS":
        yticklabels = ["Wide awake", "Sleep onset soon"]
# ax.set_yticks(yticks, minor=True)
# ax.set_yticks(yticks_major)
# ax.set_yticklabels(yticklabels)
# pvalues = pwise.set_index("tmr_condition").loc[cue_order, "p-unc"]
# for i, p in enumerate(pvalues):
#     text = fr"$p={p:.02f}$"
#     ax.text(i, 0.9, text, transform=ax.get_xaxis_transform())

ax.grid(False)
ax.tick_params(top=False, right=False)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left"]].set_position(("outward", 10))
ax.margins(x=0.1, y=0.2)

utils.export_mpl(export_path)
