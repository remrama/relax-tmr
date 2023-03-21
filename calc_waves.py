"""Detect slow-waves and generate slow-wave parameters for a single subject."""
import argparse
from pathlib import Path

from bids import BIDSLayout
import mne
import pandas as pd
import yasa

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
participant_id = f"sub-{participant:03d}"
task_id = "task-sleep"
# Drop to the only channels needed for slow waving.
eeg_channels = ["AFz", "Fz", "Fp1", "Fp2", "F3", "F4", "Cz", "C3", "C4"]


bids_root = Path(utils.config["bids_root"])
eeg_path = bids_root / participant_id / "eeg" / f"{participant_id}_{task_id}_eeg.edf"
hypno_path = bids_root / "derivatives" / participant_id / f"{participant_id}_{task_id}_hypno.tsv"
export_path_swaves = bids_root / "derivatives" / participant_id / f"{participant_id}_{task_id}_waves.tsv"
export_path_swsync = bids_root / "derivatives" / participant_id / f"{participant_id}_{task_id}_sync.tsv"

hypno = pd.read_csv(hypno_path, sep="\t")
hypno_int = hypno["value"].to_numpy()

assert hypno["duration"].nunique() == 1
epoch_length = hypno["duration"].unique()[0]
epoch_sfreq = 1 / epoch_length


# Load raw data.
raw = mne.io.read_raw_edf(eeg_path, include=eeg_channels, preload=True)

# Upsample the hynpogram to match sample rate of EEG file.
hypno_int_up = yasa.hypno_upsample_to_data(hypno_int, epoch_sfreq, raw)

swd = yasa.sw_detect(
    raw,
    hypno=hypno_int_up,
    include=(0, 1, 2, 3, 4),
    freq_sw=(0.3, 1.5),
    dur_neg=(0.3, 1.5),
    dur_pos=(0.1, 1),
    amp_neg=(40, 200),
    amp_pos=(10, 150),
    amp_ptp=(75, 350),
    coupling=True,
    coupling_params={"freq_sp": (12, 16), "p": 0.05, "time": 1},
    remove_outliers=False,
)

swp = swd.summary()
swsync = swd.get_sync_events()

swp_sidecar = {
    "Start": {
        "Description": "Start time of each detected slow-wave, in seconds from the beginning of data."
    },
    "NegPeak": {
        "LongName": "Negative Peak Location",
        "Description": "Location of the negative peak (in seconds)"
    },
    "MidCrossing": {
        "LongName": "Middle Crossing Location",
        "Description": "Location of the negative-to-positive zero-crossing (in seconds)"
    },
    "Pospeak": {
        "LongName": "Positive Peak Location",
        "Description": "Location of the positive peak (in seconds)"
    },
    "End": {
        "Description": "End time (in seconds)"
    },
    "Duration": {
        "Description": "Duration (in seconds)"
    },
    "ValNegPeak": {
        "LongName": "Negative Peak Amplitude",
        "Description": "Amplitude of the negative peak (in uV, calculated on the freq_sw bandpass-filtered signal)"
    },
    "ValPosPeak": {
        "LongName": "Positive Peak Amplitude",
        "Description": "Amplitude of the positive peak (in uV, calculated on the freq_sw bandpass-filtered signal"
    },
    "PTP": {
        "LongName": "Peak-to-peak Amplitude",
        "Description": "Peak-to-peak amplitude (= ValPosPeak - ValNegPeak, calculated on the freq_sw bandpass-filtered signal)"
    },
    "Slope": {
        "Description": "Slope between NegPeak and MidCrossing (in uV/sec, calculated on the freq_sw bandpass-filtered signal)"
    },
    "Frequency": {
        "Description": "Frequency of the slow-wave (= 1 / Duration)"
    },
    "SigmaPeak": {
        "Description": "Location of the sigma peak amplitude within a 2-sec epoch centered around the negative peak of the slow-wave. This is only calculated when coupling=True"
    },
    "PhaseAtSigmaPeak": {
        "Description": "SW phase at max sigma amplitude within a 2-sec epoch centered around the negative peak of the slow-wave. This is only calculated when coupling=True"
    },
    "ndPAC": {
        "Description": "Normalized direct PAC within a 2-sec epoch centered around the negative peak of the slow-wave. This is only calculated when coupling=True"
    },
    "Stage": {
        "LongName": "Sleep Stage",
        "Description": "Sleep stage (only if hypno was provided)"
    },
    "Channel": {
        "Description": "EEG channel name"
    },
    "IdxChannel": {
        "Description": "Index of EEG channel name"
    }
}

swsync_sidecar = {
    "Time": {
        "LongName": "Time",
        "Description": "TBD."
    },
    "Event": {
        "LongName": "Event",
        "Description": "TBD."
    },
    "Amplitude": {
        "LongName": "Amplitude",
        "Description": "TBD."
    },
    "Stage": {
        "LongName": "Stage",
        "Description": "TBD."
    },
    "Stage": {
        "LongName": "Sleep Stage",
        "Description": "Sleep stage (only if hypno was provided)"
    },
    "Channel": {
        "Description": "EEG channel name"
    },
    "IdxChannel": {
        "Description": "Index of EEG channel name"
    }
}

# Export.
utils.export_tsv(swp, export_path_swaves, index=False)
utils.export_json(swp_sidecar, export_path_swaves.with_suffix(".json"))
utils.export_tsv(swsync, export_path_swsync, index=False)
utils.export_json(swsync_sidecar, export_path_swsync.with_suffix(".json"))


# sw = yasa.sw_detect(data, sf,
    # hypno=hypno, include=(2, 3), freq_sw=(0.3, 1.5),
    # dur_neg=(0.3, 1.5), dur_pos=(0.1, 1), amp_neg=(40, 200),
    # amp_pos=(10, 150), amp_ptp=(75, 350), coupling=False,
    # remove_outliers=False, verbose=False)

# # ax = sp.plot_average(center='Peak', time_before=0.8, time_after=0.8, filt=(12, 16), ci=None)
# # df_sync = sp.get_sync_events(center='Peak', time_before=0.8, time_after=0.8)
# # coincidence = sp.get_coincidence_matrix()
# sp.summary().to_csv(export_path, sep="\t", index=False)


# hypno_events = {
#     "onset": [ epoch_length*i for i in range(n_epochs) ],
#     "duration": [ epoch_length for i in range(n_epochs) ],
#     "value" : hypno_int,
#     "description" : hypno_str,
# }
# hypno_df = pd.DataFrame.from_dict(hypno_events).join(hypno_proba.reset_index())

# proba_sidecar_info = {
#     f"proba_{x}": dict(LongName=f"Probability of {x}", Description=f"YASA's estimation of {x} likelihood")
#         for x in ["N1", "N2", "N3", "R", "W"]
# }
# proba_sidecar_info["ScorerInfo"] = {
#     "ScorerName": "YASA",
#     "EpochLength": 30,
#     "ChannelsUsed": {
#         "EEG": [eeg_channel],
#         "EOG": [eog_channel],
#         "EMG": [emg_channel],
#     }
# }
# hypno_sidecar = sleeb.eeg_events_sidecar(**proba_sidecar_info)


# # Export.
# dmlab.io.export_dataframe(hypno_df, export_path_hypno, decimals=2, mode=write_mode)
# dmlab.io.export_json(hypno_sidecar, export_path_hypno.with_suffix(".json"), mode=write_mode)

