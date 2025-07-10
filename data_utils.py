import numpy as np
import scipy.io as sio
import torch
import os

from Read_data import Read_Data

def truncate_data(data, transpose=False):

    data = data[:, 375:, :]
    if transpose:
        return data.transpose(0, 2, 1)
    else:
        return data

def load_data_2a_npz(data, label, subject_list, data_type, transpose=False):
    data_list, label_list = [], []
    for subject_num in subject_list:
        sj = Read_Data(data, label, Type=data_type, subject_num=subject_num)
        X = np.nan_to_num(sj.stack_trial_in_ch(list(range(22))))
        X = truncate_data(X, transpose=transpose)
        y = sj.Load_truelabel()
        data_list.append(X)
        label_list.append(y)
    return np.concatenate(data_list, axis=0), np.concatenate(label_list, axis=0)


def load_data_2a(subject_list, data_type,
                 crop=(375, 1500),          # 1.5 s → 6 s  ⇒ 1125 samples
                 eeg_only=True):
    """
    BCI Competition IV – data set 2a
    -------------------------------------------------------------
    Returns
        X : (n_trials, 22, crop_len)  float64
        y : (n_trials,)               int  (0-3)
    Notes
    * `crop` is given **relative to the cue-onset** in samples (@ 250 Hz).
      The default (1.5-6 s) cleanly covers the entire imagery period.
    * No trial is skipped unless `cue+crop[1]` would over-run the file.
    """

    root = f"BCICIV2a/{data_type}"          # e.g.  …/E for evaluation
    X_buf, y_buf = [], []

    for subj in subject_list:
        f = os.path.join(root, f"A{subj:02d}{data_type}.mat")   # A01E.mat
        if not os.path.isfile(f):
            print(f"[WARN] {f} missing – skipped");  continue

        mat  = sio.loadmat(f, squeeze_me=True, struct_as_record=False)
        runs = np.atleast_1d(mat["data"])                   # 9 structs

        for run in runs:
            if np.size(run.trial) == 0:                     # calibration
                continue

            X_cont = np.asarray(run.X, dtype=np.float32)    # (N,25)

            if eeg_only:
                X_cont = X_cont[:, :22]                     # drop 3 EOG

            starts = np.asarray(run.trial, dtype=int)

            labels = np.asarray(run.y,     dtype=int) - 1   # 0–3

            for idx, s0 in enumerate(starts):

                s1 = s0 + crop[1]

                if s1 > X_cont.shape[0]:                    # truncated
                    continue

                seg = X_cont[s0 + crop[0]: s1].T            # (22,1125)
                X_buf.append(seg)
                y_buf.append(labels[idx])

    if not X_buf:
        raise RuntimeError("no trials collected – check paths/params")

    X = np.stack(X_buf)             # (N,22,1125)
    y = np.asarray(y_buf)

    return X, y


def load_data_2b(subject_list, data_type, trial_length=2126):
    """
    Load and segment EEG data from .mat files for multiple subjects, combining all sessions into one array per subject.

    Parameters:
    - subject_list: List of subject IDs (e.g., [1, 2, ..., 9]).
    - data_type: "T" for training, "E" for evaluation.
    - trial_length: Number of time points per trial (default: 1000, i.e., 4 seconds at 250 Hz).

    Returns:
    - X_all: Combined EEG data across all subjects and sessions (6, trial_length, total_trials).
    - y_all: One-hot encoded labels across all subjects and sessions (total_trials, 2).
    """
    # Base path to the data directory
    base_path = "BCICIV2b/"
    data_dir = os.path.join(base_path, data_type)

    # Lists to store data across all subjects
    X_list = []
    y_list = []

    # Iterate over each subject
    for subject in subject_list:
        # Construct file path
        PATH = os.path.join(data_dir, f"B0{subject}{data_type}.mat")

        # Check if file exists
        if not os.path.exists(PATH):
            print(f"Warning: File {PATH} not found, skipping subject {subject}.")
            continue

        # Load the .mat file
        data = sio.loadmat(PATH)

        # Extract the 'data' key (array of sessions)
        sessions = data['data'][0]

        # Lists to store data for this subject across all sessions
        X_subject_list = []
        y_subject_list = []

        # Process each session for the subject
        for session_idx, session in enumerate(sessions):
            # Extract continuous EEG data (n_samples, 6)
            X_continuous = session['X'][0, 0]

            # Extract trial start indices (n_trials,)
            trial_starts = session['trial'][0, 0].flatten()

            # Extract labels (n_trials,)
            y = session['y'][0, 0].flatten() - 1  # Convert 1,2 to 0,1 for one-hot encoding

            # Number of trials in this session
            n_trials = len(trial_starts)

            # Initialize array for segmented trials (6, trial_length, n_trials)
            X_session = np.zeros((6, trial_length, n_trials))

            # Segment the continuous data into trials
            valid_trials = 0
            for i, start in enumerate(trial_starts):
                end = start + trial_length
                if end <= X_continuous.shape[0]:
                    X_session[:, :, valid_trials] = X_continuous[start:end, :].T
                    valid_trials += 1
                else:
                    print(
                        f"Warning: Trial {i} in session {session_idx} (subject {subject}) exceeds data length, skipping.")

            # Trim X_session and y to only include valid trials
            if valid_trials < n_trials:
                X_session = X_session[:, :, :valid_trials]
                y = y[:valid_trials]

            # Append to subject lists
            X_subject_list.append(X_session)
            y_subject_list.append(y)

        # Combine all trials for this subject across sessions
        X_subject = np.concatenate(X_subject_list, axis=2)  # Shape: (6, trial_length, total_trials_for_subject)
        y_subject = np.concatenate(y_subject_list)  # Shape: (total_trials_for_subject,)

        # Append to overall lists
        X_list.append(X_subject)
        y_list.append(y_subject)

    # Combine all trials across subjects
    if not X_list:
        raise ValueError("No valid trials found for the given subjects and data type.")

    X_all = np.concatenate(X_list, axis=2)[0:3, 750:1875, ...].transpose([2, 0, 1])
    y_all = np.concatenate(y_list)  # Shape: (total_trials,)

    return X_all, y_all

def check_for_nan_hook(module, output):
    if isinstance(output, tuple):
        for elem in output:
            if torch.isnan(elem).any():
                print(f"NaN found in {module} output (tuple element)")
                break
    else:
        if torch.isnan(output).any():
            print(f"NaN found in {module} output")

if __name__ == '__main__':
    data_type = "E"
    subject_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    X, y = load_data_2a(subject_list=subject_list, data_type=data_type)
    print(X.shape, y.shape)  # Shape: (6, 1000, total_trials)
    print(y)
