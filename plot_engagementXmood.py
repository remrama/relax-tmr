"""Plot relationship between relaxation task engagement and morning mood."""
import argparse
from pathlib import Path
import textwrap

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

xcolumn_choices = [
    "Difficult", # how difficult did you find the relaxation task? 1-7
    "Distracted", # how distracted were you while completing the relaxation task? 1-7
    "Enjoyable", # how enjoyable did you find the relaxation task? 1-7
    "Motivated", # how motivated were you to do well on the relaxation task? 1-7
    "Relaxed", # how relaxed were you while completing the relaxation task? 1-7
    
    "TAS",
]

ycolumn_choices = [
    "Arousal_1", # rate your current level of Arousal (slider)
    "Pleasure_1", # rate your current level of Pleasure (slider)

    "Alertness_1", # How alert are you? 0-100
    "Alertness_2", # How refreshed do you feel? 0-100
    "Alertness_3", # How well are you able to concentrate? 0-100
    "SSS", # pick what best represents how you are feeling right now (7-1)

    "EasytoSleep", # How easy was it to fall asleep? 1-5
    "SleepOverall", # How was your sleep? 1-5
    "SleepQuality", # How would you rate your sleep quality overall from last night? 4-1
    "EasytoWake", # How easy was it to wake up? 1-5
    "SleepCalm", # Was your sleep calm? 1-5
    "WellRested", # Do you feel well-rested? 1-3
    "EnoughSleep", # Did you get enough sleep? 1-5

    "PANAS_neg",
    "PANAS_pos",
    "STAI",
]

parser = argparse.ArgumentParser()
parser.add_argument("-x", "--xcolumn", type=str, default="Enjoyable", choices=xcolumn_choices)
parser.add_argument("-y", "--ycolumn", type=str, default="Pleasure_1", choices=ycolumn_choices)
parser.add_argument("-d", "--diffs", action="store_true")
args = parser.parse_args()

xcolumn = args.xcolumn
ycolumn = args.ycolumn
use_diffs = args.diffs

root_dir = Path(utils.config["bids_root"])
import_path = root_dir / "phenotype" / "debriefing.tsv"
if use_diffs:
    export_path = root_dir / "derivatives" / f"engagementXmood-{ycolumn}DIFFX{xcolumn}.png"
else:
    export_path = root_dir / "derivatives" / f"engagementXmood-{ycolumn}X{xcolumn}.png"

# layout = BIDSLayout(root_dir, validate=False)
# bids_file = layout.get(suffix="debriefing", extension="tsv")[0]
# df = bids_file.get_df()
df = pd.read_csv(import_path, index_col="participant_id", sep="\t")
meta = utils.import_json(import_path.with_suffix(".json"))

participants = utils.load_participants_file()
df = df.join(participants)

for survey in ["PANAS", "STAI", "TAS"]:
    df = utils.agg_questionnaire_columns(df, survey, delete_cols=True)

if use_diffs:
    diffs, _ = utils.load_prepost_survey_diffs()
    # diffs = diffs.join(participants["tmr_condition"])
    df[ycolumn] = diffs[ycolumn]


figsize = (1.8, 1.8)
corr_method = "spearman"
scatter_kwargs = {
    "s": 50,
    "linewidths": 0.5,
    "edgecolors": "white",
    "clip_on": False,
    "zorder": 4,
    "alpha": 0.8,
}
plot_kwargs = {
    "linewidth": 2,
}

fig, ax = plt.subplots(figsize=figsize)

for cue, cue_df in df.groupby("tmr_condition"):
    cue_color = cue_palette[cue]
    xvals = cue_df[xcolumn].to_numpy()
    yvals = cue_df[ycolumn].to_numpy()

    stat = pg.corr(xvals, yvals, method=corr_method)
    r, p = stat.loc[corr_method, ["r", "p-val"]]
    cue_df["color"] = cue_df.index.map(participant_palette)
    ax.scatter(xcolumn, ycolumn, c="color", data=cue_df, **scatter_kwargs)

    # Regression line.
    coef = np.polyfit(xvals, yvals, 1)
    poly1d_func = np.poly1d(coef)
    # ax.plot(x, y, "ko", ms=8, alpha=0.4)
    ax.plot(xvals, poly1d_func(xvals), "-", color=cue_color, label=cue_labels[cue], **plot_kwargs)

    ytext = 0.9 if cue == "relax" else 0.8
    xtext = 0.4
    text = fr"$r={r:.02f}$".replace("0.", ".")
    if p < 0.05:
        asterisks = "*" * sum([ p<cutoff for cutoff in (0.05, 0.01, 0.001) ])
        text = asterisks + text
    elif p < 0.1:
        text = "^" + text
    ax.text(xtext, ytext, text, color=cue_color, transform=ax.transAxes,
        ha="right", va="center",
    )

# Aesthetics
ax.margins(0.4)
ax.grid(False)

if ycolumn == "Arousal_1":
    ylabel = "Morning affective arousal\n" + r"Tired $\leftarrow$     $\rightarrow$ Aroused"
elif ycolumn == "Pleasure_1":
    ylabel = "Morning affective pleasure\n" + r"Negative $\leftarrow$     $\rightarrow$ Positive"
else:
    try:
        ylabel = meta[ycolumn]["Probe"]
    except:
        ylabel = ycolumn
ax.set_ylabel(ylabel)

if use_diffs:
    ylabel = r"$\Delta$ " + ylabel 
xlabel = meta[xcolumn]["Probe"]
xlabel = xlabel.replace("Overall, ", "").capitalize()
xlabel = textwrap.fill(xlabel, 25)
xticks_minor = [int(x) for x in meta[xcolumn]["Levels"]]
xticks_major = [min(xticks_minor), max(xticks_minor)]
xticklabels = [x.split("\n")[0] for x in meta[xcolumn]["Levels"].values() if "\n" in x]
xticklabels = [x.split()[1] for x in xticklabels]
ax.set_xlabel(xlabel)
ax.set_xticks(xticks_minor, minor=True)
ax.set_xticks(xticks_major)
ax.set_xticklabels(xticklabels)


# ax.legend(loc="lower left")
# ax.tick_params(top=False, bottom=False)
# if column.startswith("Alertness"):
#     ax.set_ybound(upper=100)
# ax.set_ylabel(meta[column]["Probe"])
# ax.set_xlabel("TMR condition")

# Export.
utils.export_mpl(export_path)
