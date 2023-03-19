"""Did they hear sounds during sleep? Draw a bland plot to visually check they didn't."""
from pathlib import Path

from bids import BIDSLayout
import matplotlib.pyplot as plt
import pandas as pd

import utils


utils.set_matplotlib_style("technical")

bids_root = Path(utils.config["bids_root"])
layout = BIDSLayout(bids_root, validate=False)
# bids_file = layout.get(subject=participant, task="sleep", suffix="eeg", extension="fif")[0]
# pattern = "derivatives/sub-{subject}/sub-{subject}_task-{task}_hypno.tsv"
# import_path = bids_root / "phenotype" / "debriefing.tsv"
export_path = bids_root / "derivatives" / "soundcheck.png"
bids_file = layout.get(suffix="debriefing", extension="tsv")[0]
df = bids_file.get_df()

# Did they hear any sounds?
# HeardSound_1 - No sound
# HeardSound_2 - Bell
# HeardSound_3 - Harp
# HeardSound_4 - Other
# HeardSound_4_TEXT - Other text
columns = [c for c in df if c.startswith("HeardSound") and not c.endswith("TEXT")]
table = df.set_index("participant_id")[columns].fillna(0).sort_index(axis=0).sort_index(axis=1)

figsize = (2.5, 3)
xlabel = "Did you hear any of these sounds\npresented during the night?"
ylabel = "Participants"
xticklabels = [bids_file.entities[c]["Levels"]["1"] for c in columns]
yticklabels = table.index.get_level_values("participant_id")
colormap = "binary"

fig, ax = plt.subplots(figsize=figsize)
ax.imshow(table, cmap=colormap, aspect="auto")
ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)
ax.set_xticks(range(len(xticklabels)))
ax.set_yticks(range(len(yticklabels)))
ax.set_xticklabels(xticklabels)
ax.set_yticklabels(yticklabels)
# ax.xaxis.set_major_formatter(plt.matplotlib.ticker.NullFormatter())
ax.tick_params(direction="out", top=False, right=False)
ax.grid(False)

# Export.
utils.export_mpl(export_path)
