"""Plot a single sleep statistic."""
import argparse
from pathlib import Path

from bids import BIDSLayout
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import utils


utils.set_matplotlib_style()

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--sstat", type=str, default="SME")
args = parser.parse_args()

sstat = args.sstat

bids_root = Path(utils.config["bids_root"])
export_path = bids_root / "derivatives" / f"sleepstats-{sstat}.png"

layout = BIDSLayout(bids_root, validate=False, derivatives=True)

bids_files = layout.get(task="sleep", suffix="sstats", extension="tsv")

df = pd.concat(
    [bf.get_df().assign(participant_id="sub-" + str(bf.entities["subject"])) for bf in bids_files],
    ignore_index=True,
)

cue_palette = dict(relax="mediumpurple", story="forestgreen")

participants = utils.load_participants_file()
df = df.set_index("participant_id").join(participants["tmr_condition"]).reset_index()
df = df.pivot(index="participant_id", columns="sleep_statistic", values="value")
df = df.join(participants["tmr_condition"])

x_order = ["relax", "story"]

desc = df.groupby("tmr_condition")[sstat].agg(["mean", "sem"])
desc["xval"] = desc.index.map(x_order.index)
desc["color"] = desc.index.map(cue_palette)

fig, ax = plt.subplots(figsize=(2.5, 3))

bar_kwargs = {
    "width": 0.8,
    "edgecolor": "black",
    "linewidth": 1,
    "zorder": 1,
    "error_kw": dict(capsize=3, capthick=1, ecolor="black", elinewidth=1, zorder=2),
}

bars = ax.bar(data=desc, x="xval", height="mean", yerr="sem", color="color", **bar_kwargs)
bars.errorbar.lines[2][0].set_capstyle("round")

ax.set_xticks(range(len(x_order)))
ax.set_xticklabels(x_order)
ax.margins(x=0.2)
ax.set_ybound(upper=100)
ax.tick_params(top=False, bottom=False, right=False)
ax.set_ylabel(sstat)
ax.set_xlabel("TMR condition")

# Export.
utils.export_mpl(export_path)
