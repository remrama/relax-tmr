"""Helper functions (also configures MNE logging)."""
from datetime import timezone
import json
from pathlib import Path

import colorcet as cc
import matplotlib.pyplot as plt
import mne
import pandas as pd


def import_json(filepath: str, **kwargs) -> dict:
    """Loads json file as a dictionary"""
    with open(filepath, "rt", encoding="utf-8") as fp:
        return json.load(fp, **kwargs)

def export_json(obj: dict, filepath: str, mode: str="wt", **kwargs):
    kwargs = {"indent": 4} | kwargs
    with open(filepath, mode, encoding="utf-8") as fp:
        json.dump(obj, fp, **kwargs)

def export_tsv(df, filepath, mkdir=True, **kwargs):
    kwargs = {"sep": "\t", "na_rep": "n/a"} | kwargs
    if mkdir:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, **kwargs)

def export_mpl(filepath, mkdir=True, close=True):
    filepath = Path(filepath)
    if mkdir:
        filepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath)
    # plt.savefig(filepath.with_suffix(".pdf"))
    if close:
        plt.close()

# Load configuration file so it's accessible from utils
config = import_json("./config.json")

# Configure MNE logging here so it's the same across all files.
mne_logfile = Path(config["bids_root"]) / "derivatives" / "mne.log"
mne_verbosity = config["mne_verbosity"]
mne.set_log_level(verbose=mne_verbosity)
mne.set_log_file(fname=mne_logfile, overwrite=False)


def load_participants_file():
    bids_root = Path(config["bids_root"])
    filepath = bids_root / "participants.tsv"
    date_columns = ["bedtime", "waketime"]
    df = pd.read_csv(filepath, parse_dates=date_columns, index_col="participant_id", sep="\t")
    for dcol in date_columns:
        df[dcol] = df[dcol].dt.tz_localize("US/Eastern").dt.tz_convert(timezone.utc)
    # layout = BIDSLayout(bids_root)
    # bf = layout.get(suffix="participants", extension=".tsv")[0]
    # df = bf.get_df(index_col="participant_id")
    return df

def load_participant_palette():
    # df = load_participants_file().reset_index()
    # df["color"] = df.index.map(cc.cm.glasbey_dark)
    # palette = df.set_index("participant_id")["color"].to_dict()
    pp = load_participants_file()
    relax = pp.query("tmr_condition == 'relax'").index.tolist()
    story = pp.query("tmr_condition == 'story'").index.tolist()
    relax_palette = {p: cc.cm.glasbey_cool(i) for i, p in enumerate(relax)}
    story_palette = {p: cc.cm.glasbey_warm(i) for i, p in enumerate(story)}
    return relax_palette | story_palette

def participant_values():
    """Return a list of all valid participant values as integers. (sub-001 == key-value)"""
    return load_participants_file().index.str.split("-").str[-1].astype(int).tolist()


def imputed_sum(row):
    """Return nan if more than half of responses are missing."""
    return np.nan if row.isna().mean() > 0.5 else row.fillna(row.mean()).sum()

def agg_questionnaire_columns(df, questionnaire_name, delete_cols=False):
    columns = [ c for c in df if c.startswith(f"{questionnaire_name}_") ]
    assert columns
    if questionnaire_name == "PANAS":
        positive_probes = [1, 3, 5, 9, 10, 12, 14, 16, 17, 19]
        positive_columns = [ c for c in columns if int(c.split("_")[-1]) in positive_probes ]
        negative_columns = [ c for c in columns if c not in positive_columns ]
        df["PANAS_pos"] = df[positive_columns].apply(imputed_sum, axis=1)
        df["PANAS_neg"] = df[negative_columns].apply(imputed_sum, axis=1)
        if delete_cols:
            df = df.drop(columns=positive_columns + negative_columns)
    elif questionnaire_name in ["STAI", "TAS"]:
        df[questionnaire_name] = df[columns].apply(imputed_sum, axis=1)
        if delete_cols:
            df = df.drop(columns=columns)
    return df


def load_prepost_survey_diffs():
    root_dir = Path(config["bids_root"])
    import_path_pre = root_dir / "phenotype" / "initial.tsv"
    import_path_post = root_dir / "phenotype" / "debriefing.tsv"
    pre = pd.read_csv(import_path_pre, index_col="participant_id", sep="\t")
    post = pd.read_csv(import_path_post, index_col="participant_id", sep="\t")
    for survey in ["PANAS", "STAI"]:
        pre = agg_questionnaire_columns(pre, survey, delete_cols=True)
        post = agg_questionnaire_columns(post, survey, delete_cols=True)
    pre = pre.select_dtypes("number").dropna(axis=1)
    post = post.select_dtypes("number").dropna(axis=1)
    overlapping_cols = list(set(pre.columns) & set(post.columns))
    diff = post[overlapping_cols] - pre[overlapping_cols]
    meta = import_json(import_path_pre.with_suffix(".json"))
    meta = {k: v for k, v in meta.items() if k in overlapping_cols}
    return diff, meta


################################################################################
# GENERATING BIDS SIDECAR FILES
################################################################################


def generate_eeg_sidecar(
        task_name,
        task_description,
        task_instructions,
        reference_channel,
        ground_channel,
        sampling_frequency,
        recording_duration,
        n_eeg_channels,
        n_eog_channels,
        n_ecg_channels,
        n_emg_channels,
        n_misc_channels,
        **kwargs,
    ):
    return {
        "TaskName": task_name,
        "TaskDescription": task_description,
        "Instructions": task_instructions,
        "InstitutionName": "Northwestern University",
        "Manufacturer": "Neuroscan",
        "ManufacturersModelName": "tbd",
        "CapManufacturer": "tbd",
        "CapManufacturersModelName": "tbd",
        "PowerLineFrequency": 60,
        "EEGPlacementScheme": "10-20",
        "EEGReference": f"single electrode placed on {reference_channel}",
        "EEGGround": f"single electrode placed on {ground_channel}",
        "SamplingFrequency": sampling_frequency,
        "EEGChannelCount": n_eeg_channels,
        "EOGChannelCount": n_eog_channels,
        "ECGChannelCount": n_ecg_channels,
        "EMGChannelCount": n_emg_channels,
        "MiscChannelCount": n_misc_channels,
        "TriggerChannelCount": 0,
        "SoftwareFilters": "tbd",
        "HardwareFilters": {
            "tbd": {
                "tbd": "tbd",
                "tbd": "tbd"
            }
        },
        "RecordingType": "continuous",
        "RecordingDuration": recording_duration,
    } | kwargs

def generate_channels_sidecar(**kwargs):
    return {
        "name": "See BIDS spec",
        "type": "See BIDS spec",
        "units": "See BIDS spec",
        "description": "See BIDS spec",
        "sampling_frequency": "See BIDS spec",
        "reference": "See BIDS spec",
        "low_cutoff": "See BIDS spec",
        "high_cutoff": "See BIDS spec",
        "notch": "See BIDS spec",
        "status": "See BIDS spec",
        "status_description": "See BIDS spec",
        "RespirationHardware": "tbd", # seems like a good thing to add??
    } | kwargs

def generate_events_sidecar(columns, **kwargs):
    column_info = {
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
            "Description": "Readable explanation of value markers column",
        },
        "volume": {
            "LongName": "words",
            "Description": "more words",
        },
        "trial_type": {
            "LongName": "General event category",
            "Description": "Very different event types are included, so this clarifies",
            "Levels": {
                "tmr": "A sound cue for targeted memory reactivation",
                "misc": "Things like lights-on lights-off or note"
            }
        }
    }
    info = { c: column_info[c] for c in columns }
    # info["StimulusPresentation"] = {
    #     "OperatingSystem": "Linux Ubuntu 18.04.5",
    #     "SoftwareName": "Psychtoolbox",
    #     "SoftwareRRID": "SCR_002881",
    #     "SoftwareVersion": "3.0.14",
    #     "Code": "doi:10.5281/zenodo.3361717"
    # }
    return info | kwargs


################################################################################
# PLOTTING
################################################################################


def cmap2hex(cmap, n_intervals) -> list:
    if isinstance(cmap, str):
        if (cmap := cc.cm.get(cmap)) is None:
            try:
                cmap = plt.get_cmap(cmap)
            except ValueError as e:
                raise e
    assert isinstance(cmap, plt.matplotlib.colors.LinearSegmentedColormap)
    stops = [ 0 + x*1/(n_intervals-1) for x in range(n_intervals) ] # np.linspace
    hex_codes = []
    for s in stops:
        assert isinstance(s, float)
        rgb_floats = cmap(s)
        rgb_ints = [ round(f*255) for f in rgb_floats ]
        hex_code = "#{0:02x}{1:02x}{2:02x}".format(*rgb_ints)
        hex_codes.append(hex_code)
    return hex_codes

def set_matplotlib_style(mpl_style="technical"):
    if mpl_style == "technical":
        plt.rcParams["savefig.dpi"] = 600
        plt.rcParams["interactive"] = True
        plt.rcParams["figure.constrained_layout.use"] = True
        plt.rcParams["font.family"] = "Times New Roman"
        # plt.rcParams["font.sans-serif"] = "Arial"
        plt.rcParams["mathtext.fontset"] = "custom"
        plt.rcParams["mathtext.rm"] = "Times New Roman"
        plt.rcParams["mathtext.cal"] = "Times New Roman"
        plt.rcParams["mathtext.it"] = "Times New Roman:italic"
        plt.rcParams["mathtext.bf"] = "Times New Roman:bold"
        plt.rcParams["font.size"] = 8
        plt.rcParams["axes.titlesize"] = 8
        plt.rcParams["axes.labelsize"] = 8
        plt.rcParams["axes.labelsize"] = 8
        plt.rcParams["xtick.labelsize"] = 8
        plt.rcParams["ytick.labelsize"] = 8
        plt.rcParams["axes.linewidth"] = 0.8 # edge line width
        plt.rcParams["axes.axisbelow"] = True
        plt.rcParams["axes.grid"] = True
        plt.rcParams["axes.grid.axis"] = "y"
        plt.rcParams["axes.grid.which"] = "major"
        plt.rcParams["axes.labelpad"] = 4
        plt.rcParams["xtick.top"] = True
        plt.rcParams["ytick.right"] = True
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"
        plt.rcParams["grid.color"] = "gainsboro"
        plt.rcParams["grid.linewidth"] = 1
        plt.rcParams["grid.alpha"] = 1
        plt.rcParams["legend.frameon"] = False
        plt.rcParams["legend.edgecolor"] = "black"
        plt.rcParams["legend.fontsize"] = 8
        plt.rcParams["legend.title_fontsize"] = 8
        plt.rcParams["legend.borderpad"] = .4
        plt.rcParams["legend.labelspacing"] = .2 # the vertical space between the legend entries
        plt.rcParams["legend.handlelength"] = 2 # the length of the legend lines
        plt.rcParams["legend.handleheight"] = .7 # the height of the legend handle
        plt.rcParams["legend.handletextpad"] = .2 # the space between the legend line and legend text
        plt.rcParams["legend.borderaxespad"] = .5 # the border between the axes and legend edge
        plt.rcParams["legend.columnspacing"] = 1 # the space between the legend line and legend text
    else:
        raise ValueError(f"matplotlib style {mpl_style} is not an option")