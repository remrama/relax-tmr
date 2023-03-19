"""Memory/venice scores split by TMR condition."""
from pathlib import Path

from bids import BIDSLayout
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import utils


utils.set_matplotlib_style()


bids_root = Path(utils.config["bids_root"])
layout = BIDSLayout(bids_root, validate=False)
export_path_data = bids_root / "derivatives" / "venice.tsv"
export_path_plot = export_path_data.with_suffix(".png")

# Load data
bf = layout.get(suffix="debriefing", extension="tsv")[0]
df = bf.get_df(index_col="participant_id")
meta = bf.get_metadata()

venice_columns = [c for c in df if c.startswith("Venice_")]
multchoice_cols = [vc for vc in venice_columns if "MC" in vc]
freerecall_cols = [vc for vc in venice_columns if "MC" not in vc]
# # Replace multiple choice responses with text options.
# answers = {
#     1: "The Adriatic Sea",  # (8)
#     2: "Roman refugees",  # (8)
#     3: "Wood",  # (8)
#     4: "270,000",  # (8)
#     5: "The extraction of water from artesian wells disrupting the sediment below the houses",  # (8)
#     6: "Byzantine empire",  # (8)
# }
# for c in multchoice_cols:
#     mapping = { int(k): v for k, v in meta[c]["Levels"].items() }
#     df[c] = df[c].map(mapping)
df[freerecall_cols].dropna(how="all").T
venice = df[multchoice_cols].dropna().eq(8)
venice["pct_correct"] = venice.mean(axis=1)

pp = utils.load_participants_file()
palette = utils.load_participant_palette()

res = venice.join(pp).reset_index()

x_order = ["relax", "story"]

# Draw.
fig, ax = plt.subplots(figsize=(2, 3))
data_kwargs = dict(data=res, x="tmr_condition", y="pct_correct", order=x_order)
sns.boxplot(color="white", saturation=1, linewidth=1, ax=ax, **data_kwargs)
# sns.barplot(color="gainsboro", ax=ax, **data_kwargs)
sns.swarmplot(
    hue="participant_id",
    palette=palette,
    linewidth=1,
    edgecolor="white",
    legend=False,
    clip_on=False,
    ax=ax,
    **data_kwargs,
)

# Aesthetics.
ax.set_ylim(0, 1)
ax.set_ylabel("Venice percent correct")
ax.set_xlabel("Cue condition")
ax.tick_params(top=False, right=False)
ax.spines[["top", "right"]].set_visible(False)
ax.spines["left"].set_position(("outward", 5))

# Export.
utils.export_tsv(venice, export_path_data, index=True)
utils.export_mpl(export_path_plot)
