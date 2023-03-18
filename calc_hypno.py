"""Generate an events-style hypnogram for the entire night (of one participant)."""
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


################################################################################
# SETUP
################################################################################

participant = args.participant  # from command-line
participant_id = f"sub-{participant:03d}"
task_id = "task-sleep"
eeg_channel = "C4"
eog_channel = "EOG1"
emg_channel = "EMG1"
epoch_length = 30

bids_root = Path(utils.config["bids_root"])

screening_path = bids_root / "phenotype" / "screening.tsv"
eeg_path = bids_root / participant_id / "eeg" / f"{participant_id}_{task_id}_eeg.edf"
export_path = bids_root / "derivatives" / participant_id / f"{participant_id}_{task_id}_hypno.tsv"

# layout = BIDSLayout(bids_root, validate=False)
# # stimuli_dir = bids_root / "stimuli"
# bids_file = layout.get(subject=participant, task="sleep", suffix="eeg", extension="fif")[0]
# pattern = "derivatives/sub-{subject}/sub-{subject}_task-{task}_hypno.tsv"
# export_path = Path(layout.build_path(bids_file.entities, pattern, validate=False))

# Load demographic information.
# screening = pd.read_csv(screening_path, index_col="SID", sep="\t")
# age = screening.loc[participant, "Age"]
# sex = screening.loc[participant, "Sex"]
# metadata = dict(age=age, male=sex)
metadata = None

# Load raw data.
raw = mne.io.read_raw_edf(eeg_path, include=[eeg_channel, eog_channel, emg_channel], preload=True)
# raw.resample(100)
# raw.filter(0.1, 40, filter_length="auto", method="fir")

# Perform YASA's automatic sleep staging.
sls = yasa.SleepStaging(
    raw,
    eeg_name=eeg_channel,
    eog_name=eog_channel,
    emg_name=emg_channel,
    metadata=metadata,
)

hypno_str = sls.predict()
hypno_proba = sls.predict_proba().add_prefix("proba_")

# Generate events dataframe for hypnogram.
n_epochs = len(hypno_str)
hypno_int = yasa.hypno_str_to_int(hypno_str)
hypno_events = {
    "onset": [epoch_length*i for i in range(n_epochs)],
    "duration": epoch_length,
    "value" : hypno_int,
    "description" : hypno_str,
    "scorer": f"YASA-v{yasa.__version__}",
    "eeg_channel": eeg_channel,
    "eog_channel": eog_channel,
    "emg_channel": emg_channel,
}
hypno = pd.DataFrame.from_dict(hypno_events).join(hypno_proba.reset_index())

hypno_sidecar = {
    "onset": {
        "LongName": "Onset (in seconds) of the event",
        "Description": "Onset (in seconds) of the event"
    },
    "duration": {
        "LongName": "Duration of the event (measured from onset) in seconds",
        "Description": "Duration of the event (measured from onset) in seconds"
    },
    "value": {
        "LongName": "Marker/trigger value associated with the event",
        "Description": "Marker/trigger value associated with the event"
    },
    "description": {
        "LongName": "Value description",
        "Description": "Readable explanation of value markers column"
    },
    "scorer": {},
    "eeg_channel": {},
    "eog_channel": {},
    "emg_channel": {}
}

for x in ["N1", "N2", "N3", "R", "W"]:
    hypno_sidecar[f"proba_{x}"] = {
        "LongName": f"Probability of {x}",
        "Description": f"YASA's estimation of {x} likelihood"
    }

# Export.
utils.export_tsv(hypno, export_path, index=False)
utils.export_json(hypno_sidecar, export_path.with_suffix(".json"))
