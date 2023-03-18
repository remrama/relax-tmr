"""
Convert EEG source data (BrainVision)
to EEG BIDS-formatted file (EDF) and corresponding metadata files.

Trim sleep to lights-off --> lights-on.
Apply minimal preprocessing.
Handle trigger codes and export to events file.

=> sub-001/eeg/sub-001_ses-001_task-sleep_eeg.fif
=> sub-001/eeg/sub-001_ses-001_task-sleep_eeg.json
=> sub-001/eeg/sub-001_ses-001_task-sleep_events.tsv
=> sub-001/eeg/sub-001_ses-001_task-sleep_events.json
=> sub-001/eeg/sub-001_ses-001_task-sleep_channels.tsv
=> sub-001/eeg/sub-001_ses-001_task-sleep_channels.json
"""
import argparse
from pathlib import Path
from zoneinfo import ZoneInfo

import mne
import numpy as np
import pandas as pd
from scipy.io import loadmat

import utils



parser = argparse.ArgumentParser()
parser.add_argument(
    "-p", "--participant", type=int, required=True, choices=utils.participant_values()
)
args = parser.parse_args()


################################################################################
# SETUP
################################################################################

participant = args.participant
task = "sleep"

participant_id = f"sub-{participant:03d}"
task_id = f"task-{task}"

bids_root = Path(utils.config["bids_root"])

import_parent = bids_root / "sourcedata" / participant_id
import_name = f"RELAX_{participant:d}_Overnight.vhdr"
import_path = import_parent / import_name

def get_logfile_path(import_dir):
    potential_paths = list(import_dir.glob("*Overnight_logFile*.mat"))
    assert len(potential_paths) == 1
    return potential_paths[0]
logfile_path = get_logfile_path(import_parent)

# Pick paths for exporting.
export_parent = bids_root / participant_id / "eeg"
export_path_eeg = export_parent / f"{participant_id}_{task_id}_eeg.edf"
export_path_events = export_parent / f"{participant_id}_{task_id}_events.tsv"
export_path_channels = export_parent / f"{participant_id}_{task_id}_channels.tsv"
export_parent.mkdir(parents=True, exist_ok=True)

# Load participants file.
participants = utils.load_participants_file()
# measurement_date = participants.loc[participant_id, "measurement_date"]
reference_channel = participants.at[participant_id, "eeg_reference"]
ground_channel = participants.at[participant_id, "eeg_ground"]
notch_frequency = 60

# Load raw data.
raw = mne.io.read_raw_brainvision(
    vhdr_fname=import_path,
    eog=("EOG1", "EOG2"),
    misc="auto",
    scale=1,
    preload=False,
)


################################################################################
# Handle corrections

# Forgot to hit lights_out for some subjects.
if participant in [103, 106, 107]:
    t1 = participants.loc[participant_id, "bedtime"]
    t0 = raw.info["meas_date"].replace(tzinfo=ZoneInfo("US/Eastern")).astimezone(ZoneInfo("UTC"))
    tdelta = (t1 - t0).total_seconds()
    # raw.annotations.append(onset=tdelta, duration=0, description="lights_out")
    raw.annotations.append(onset=tdelta, duration=0, description="Comment/Lights out")

# Hit lights_out twice for sub-105
if participant == 105:
    raw.annotations.delete(9)

# Hit lights_on twice for sub-108 (note one was probably a manual comment bc lowercase)
if participant == 108:
    raw.annotations.delete(-1)

################################################################################


raw.set_channel_types({"EMG1": "emg", "EMG2": "emg"})

# Load events log.
# Load matlab log file to get better trigger code descriptions.
mat = loadmat(logfile_path)
matlog = mat["logFile"]
tmr = pd.DataFrame(matlog)
def reduce_dims(x):
    if x.size > 0:
        while isinstance(x, np.ndarray):
            x = x[0]
        return x
    else:
        return pd.NA
tmr = tmr.applymap(reduce_dims)
tmr.columns = tmr.loc[0, :].tolist()
tmr = tmr.drop(0, axis=0)
assert tmr.groupby("code")["event"].nunique().eq(1).all(), "Each code should map to unique event"


################################################################################
# CONSTRUCT EVENTS DATAFRAME
################################################################################

# Generate events DataFrame from EEG file.
## Note: Could also get there from raw.annotations.to_dataframe(), both require manipulation.
array, desc2val = mne.events_from_annotations(raw)
events = pd.DataFrame(array, columns=["onset", "duration", "value"])
val2desc = { v: k for k, v in desc2val.items() }
events["description"] = events["value"].map(val2desc)
events["onset"] /= raw.info["sfreq"]  # mne returns onset in unit of sample number, change to seconds
# events_val2descr = { v: val2txt[v] if v in val2txt else k for k, v in events_descr2val.items() }
events = events.drop(0, axis=0)  # First event is meaningless
events["duration"] = 0  # Correct default value of 0.002 which isn't meaningful.

# Convert stimulus messages to be more meaningful
stim_val2desc = tmr.set_index("code")["event"].to_dict()
stim_val2desc = { f"Stimulus/S  {k:d}": v for k, v in stim_val2desc.items() }
events["description"] = events["description"].replace(stim_val2desc)

# Remove excess text
events["description"] = events["description"].str.split("Comment/").str[-1]
events["description"] = events["description"].str.split(r"Sounds\\").str[-1]
events["description"] = events["description"].str.lower().str.replace(" ", "_")

# Remove existing annotations (to avoid redundancy).
while raw.annotations:
    raw.annotations.delete(0)

# Crop EEG and events to lights off/on.
tmin, tmax = events.set_index("description").loc[["lights_out", "lights_on"], "onset"]
raw.crop(tmin, tmax)
events = events.set_index("onset").loc[tmin:tmax].reset_index()
events["onset"] -= tmin


################################################################################
# PREPROCESSING
################################################################################

raw.load_data()

# Re-referencing
raw.set_eeg_reference("average", projection=False)

# Bandpass filtering
filter_params = dict(filter_length="auto", method="fir", fir_window="hamming")
filter_cutoffs = {  # from AASM guidelines
    "eeg": (0.3, 35), # Hz; Low-cut, High-cut
    "eog": (0.3, 35),
    "emg": (10, 100),
}
raw.filter(*filter_cutoffs["eeg"], picks="eeg", **filter_params)
raw.filter(*filter_cutoffs["eog"], picks="eog", **filter_params)
raw.filter(*filter_cutoffs["emg"], picks="emg", **filter_params)

# Downsampling
raw.resample(100)

# Anonymizing
raw.set_meas_date(None)
mne.io.anonymize_info(raw.info)




################################################################################
# GENERATE BIDS METADATA
################################################################################

# Channels DataFrame
# n_total_channels = raw.channel_count
# Convert from MNE FIFF codes
fiff2str = {2: "eeg", 202: "eog", 302: "emg", 402: "ecg", 502: "misc"}
channels_data = {
    "name": [ x["ch_name"] for x in raw.info["chs"] ], # OR raw.ch_names
    "type": [ fiff2str[x["kind"]].upper() for x in raw.info["chs"] ],
    # "types": [ raw.get_channel_types(x)[0].upper() for x in raw.ch_names ],
    "units": [ x["unit"] for x in raw.info["chs"] ],
    "description": "none",
    "sampling_frequency": raw.info["sfreq"],
    "reference": reference_channel,
    "low_cutoff": raw.info["highpass"],
    "high_cutoff": raw.info["lowpass"],
    "notch": notch_frequency,
    "status": "none",
    "status_description": "none",
}
channels = pd.DataFrame.from_dict(channels_data)
channels_sidecar = utils.generate_channels_sidecar()

# EEG sidecar
ch_type_counts = channels["type"].value_counts()
ch_type_counts = ch_type_counts.reindex(["EEG", "EOG", "EMG", "ECG", "MISC"], fill_value=0)
eeg_sidecar = utils.generate_eeg_sidecar(
    task_name=task,
    task_description="Participants went to sleep and TMR cues were played quietly during slow-wave sleep",
    task_instructions="Go to sleep.",
    reference_channel=reference_channel,
    ground_channel=ground_channel,
    sampling_frequency=raw.info["sfreq"],
    recording_duration=raw.times[-1],
    n_eeg_channels=int(ch_type_counts.at["EEG"]),
    n_eog_channels=int(ch_type_counts.at["EOG"]),
    n_ecg_channels=int(ch_type_counts.at["ECG"]),
    n_emg_channels=int(ch_type_counts.at["EMG"]),
    n_misc_channels=int(ch_type_counts.at["MISC"]),
)

# Events sidecar
events_sidecar = utils.generate_events_sidecar(events.columns)


################################################################################
# EXPORTING
################################################################################

mne.export.export_raw(export_path_eeg, raw, fmt="edf", add_ch_type=False, overwrite=True)
utils.export_json(eeg_sidecar, export_path_eeg.with_suffix(".json"))
utils.export_tsv(channels, export_path_channels)
utils.export_json(channels_sidecar, export_path_channels.with_suffix(".json"))
utils.export_tsv(events, export_path_events, index=False)
utils.export_json(events_sidecar, export_path_events.with_suffix(".json"))
