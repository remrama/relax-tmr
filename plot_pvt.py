"""Analyze PVT, pre/post analyze and plot."""
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


root_dir = Path(utils.config["bids_root"])

export_path = root_dir / "derivatives" / "pvt.png"

layout = BIDSLayout(root_dir, derivatives=False, validate=False)

bids_files = layout.get(task="pvt", suffix="beh", extension="tsv")
participants = utils.load_participants_file()

dataframe_list = []
for bf in bids_files:
    participant_id = "sub-" + bf.entities["subject"]
    acquisition_id = "acq-" + bf.entities["acquisition"]
    data = bf.get_df().assign(participant_id=participant_id, acquisition_id=acquisition_id)
    dataframe_list.append(data)

df = pd.concat(dataframe_list, ignore_index=True).set_index(["participant_id", "acquisition_id"])
df = df.join(participants["tmr_condition"]).set_index(["tmr_condition"], append=True)

### I think "bad" column combines "bp" and "fs" columns.
# Ignore false starts (only a few), nr (no response?), and bp (bad press?) for now.
# Remove bad trials so they don't influence mean.
df = df.query("bad.eq(False)").drop(columns=["bad", "bp", "fs", "nr"])
desc = df.groupby(level=["participant_id", "acquisition_id", "tmr_condition"])["rt"].describe()
    # .agg(["count", "mean", "std", "sem", "min", "median", "max"])

# df = pd.concat(
#     [bf.get_df().assign(participant_id="sub-" + str(bf.entities["subject"])) for bf in bids_files],
#     ignore_index=True,
# )

pre = desc.loc[(slice(None), "acq-pre", slice(None)), :].droplevel(["acquisition_id", "tmr_condition"])
post = desc.loc[(slice(None), "acq-post", slice(None)), :].droplevel(["acquisition_id", "tmr_condition"])
diff = (post - pre).assign(acquisition_id="diff").set_index("acquisition_id", append=True)
diff = diff.join(participants["tmr_condition"]).set_index(["tmr_condition"], append=True)


desc = pd.concat([desc, diff]).sort_index()


summ = desc.groupby(["tmr_condition", "acquisition_id"])["mean"].agg(["mean", "sem"])


x_order = ["relax", "story"]

yvals = summ.loc[(x_order, "diff"), "mean"].to_numpy()
yerrs = summ.loc[(x_order, "diff"), "sem"].to_numpy()
xvals = np.arange(yvals.size)

figsize = (2, 3)



lines_kwargs = dict(linewidths=0.5, zorder=3)
bar_kwargs = {
    "width": 0.8,
    "color": "white",
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
ax.axhline(0, color="black", linewidth=1, linestyle="dashed")
bars = ax.bar(xvals, yvals, yerr=yerrs, **bar_kwargs)
bars.errorbar.lines[2][0].set_capstyle("round")

# Draw individual participants.
df_ = desc.loc[(slice(None), "diff", slice(None)), :].reset_index()
jitter = 0.1
np.random.seed(1)
participant_palette = utils.load_participant_palette()
df_["xval"] = df_["tmr_condition"].map(lambda x: x_order.index(x))
df_["xval"] += np.random.uniform(-jitter, jitter, size=len(df_))
df_["color"] = df_["participant_id"].map(participant_palette)#.to_numpy()
ax.scatter("xval", "mean", c="color", data=df_, **scatter_kwargs)


a, b = df_.groupby("tmr_condition")["mean"].apply(list)
d = pg.compute_effsize(a, b, paired=False, eftype="cohen")
# p = wilcoxon.at["Wilcoxon", "p-val"]
# pcolor ="black" if p < 0.1 else "gainsboro"
color = "black"
ax.hlines(
    y=1.05,
    xmin=xvals[0],
    xmax=xvals[1],
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
ax.text(0.5, 1.05, text,
    color=color,
    transform=ax.transAxes,
    ha="center",
    va="bottom",
)

# Aesthetics
ax.set_xticks(xvals)
ax.set_xticklabels(x_order)
ax.margins(x=0.2, y=0.4)
ax.tick_params(top=False, bottom=False)
ax.set_ylabel(r"Post-sleep change in PVT reaction time ($\Delta$ ms)")
ax.set_xlabel("TMR condition")

# Export.
utils.export_mpl(export_path)
