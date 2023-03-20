"""Correlate cue frequency with sleep quality."""
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

parser = argparse.ArgumentParser()
parser.add_argument("-x", "--xvar", type=str, default="n_sws_cues")
parser.add_argument("-y", "--yvar", type=str, default="SME")
args = parser.parse_args()

xvar = args.xvar
yvar = args.yvar

root_dir = Path(utils.config["bids_root"])
export_path = root_dir / "derivatives" / f"corr-{xvar}X{yvar}.png"

cue_order = utils.config["cue_order"]
cue_palette = utils.config["cue_palette"]
participant_palette = utils.load_participant_palette()

df = (utils.load_participants_file()
    .join(pd.read_csv(root_dir/"derivatives"/"task-pvt.tsv", index_col="participant_id", sep="\t"))
    .join(pd.read_csv(root_dir/"derivatives"/"task-sleep_ncues.tsv", index_col="participant_id", sep="\t"))
    .join(pd.read_csv(root_dir/"derivatives"/"task-sleep_sstats.tsv", index_col="participant_id", sep="\t"))
    .join(pd.read_csv(root_dir/"phenotype"/"debriefing.tsv", index_col="participant_id", sep="\t"))
)

df["color"] = df.index.map(participant_palette)

# desc = df.groupby("tmr_condition")[].agg(["mean", "sem"])
# desc["xval"] = desc.index.map(cue_order.index)
# desc["color"] = desc.index.map(cue_palette)

# df[yvar] = df[yvar].mul(df["Enjoyable"])

g = sns.lmplot(
    data=df,
    x=xvar,
    y=yvar,
    hue="tmr_condition",
    palette=cue_palette,
)


