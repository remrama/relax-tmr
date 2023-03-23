"""Plot interaction between pre/post measure and TMR condition."""
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
desc["xval"] = desc["tmr_condition"].map(cue_order.index).multiply(3)
desc["xval"] = desc["xval"].add(desc["time"].eq("post").astype(int))
desc["color"] = desc["tmr_condition"].map(cue_palette)

bar_kwargs = {
    "width": 0.8,
    "edgecolor": "black",
    "linewidth": 1,
    "zorder": 1,
    "error_kw": dict(capsize=3, capthick=1, ecolor="black", elinewidth=1, zorder=2),
}

figsize = (3, 3)
fig, ax = plt.subplots(figsize=figsize)
bars = ax.bar(data=desc, x="xval", height="mean", yerr="sem", color="color", **bar_kwargs)
bars.errorbar.lines[2][0].set_capstyle("round")

jitter = 0.1
np.random.seed(1)
df["color"] = df["participant_id"].map(participant_palette)
df["xval"] = df["tmr_condition"].map(cue_order.index).multiply(3)
df["xval"] = df["xval"].add(df["time"].eq("post").astype(int))
df["xval"] = df["xval"].add(np.random.uniform(-jitter, jitter, size=len(df)))
scatter_kwargs = {
    "s": 30,
    "linewidths": 0.5,
    "edgecolors": "white",
    "clip_on": False,
    "zorder": 4,
}

ax.scatter(data=df, x="xval", y=column, c="color", **scatter_kwargs)


xticks = desc["xval"].sort_values().to_list()
xticklabels = desc.sort_values("xval")["time"].to_list()

ylabel = column
ax.set_ylabel(column)



pvalues = pwise.set_index("tmr_condition").loc[cue_order, "p-unc"]
for i, p in enumerate(pvalues):
    text = fr"$p={p:.02f}$"
    ax.text(i, 0.9, text, transform=ax.get_xaxis_transform())


# utils.export_mpl(export_path)
