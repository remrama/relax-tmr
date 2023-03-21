"""Event-related LFP for a single subject, for all channels, averaged over cues."""
import argparse
from pathlib import Path

from bids import BIDSLayout
import mne
import pandas as pd
import yasa

import utils


# Parse command-line arguments.
parser = argparse.ArgumentParser()
parser.add_argument(
    "-p", "--participant", type=int, required=True, choices=utils.participant_values()
)
args = parser.parse_args()

participant = args.participant  # from command-line
participant_id = f"sub-{participant:03d}"
task_id = "task-sleep"

root_dir = Path(utils.config["bids_root"])
eeg_path = root_dir / participant_id / "eeg" / f"{participant_id}_{task_id}_eeg.edf"
events_path = root_dir / participant_id / "eeg" / f"{participant_id}_{task_id}_events.tsv"
export_path = root_dir / "derivatives" / participant_id / f"{participant_id}_{task_id}_lfp.tsv"

# Load data
raw = mne.io.read_raw_edf(eeg_path)

raw.set_channel_types({"EOG1": "eog", "EOG2": "eog", "EMG1": "emg", "EMG2": "emg"})
raw.set_montage("standard_1020")

events = pd.read_csv(events_path, sep="\t")

# baseline_length = 0
event_length = 10
sfreq = raw.info["sfreq"]

events = events.query("description.ne('start_cueing.wav')")
events = events.query("description.str.endswith('.wav')")
events["onset"] = events["onset"].mul(sfreq).round().astype(int)
events_arr = events[["onset", "duration", "value"]].to_numpy()
events_dict = events.set_index("description")["value"].to_dict()

epochs = mne.Epochs(raw, events_arr, tmax=event_length, event_id=events_dict)

epochs.load_data()
psd = epochs.compute_psd(method="welch", fmin=0.5, fmax=4.5, tmin=None, tmax=None)
avg = psd.average()

data, freqs = avg.get_data(return_freqs=True)

df = pd.DataFrame(
    data,
    index=pd.Index(avg.ch_names, name="channel"),
    columns=pd.Index(freqs, name="frequency"),
)

df = df.melt(ignore_index=False, value_name="power")

utils.export_tsv(df, export_path)

