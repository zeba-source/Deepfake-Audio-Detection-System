from typing import Optional
import numpy as np
import warnings

# Suppress specific warnings that may occur during threaded execution
warnings.filterwarnings('ignore', category=UserWarning)

import librosa
from scipy.fftpack import dct


def extract_cqcc_features(file_path: str, sr: int = 16000) -> Optional[np.ndarray]:
    """Extract CQCC (Constant-Q Cepstral Coefficients) from an audio file.

    This function loads an audio file, computes the Constant-Q Transform (CQT),
    converts the magnitude to a log scale (dB), applies a Discrete Cosine Transform
    (DCT) across frequency bins to obtain cepstral coefficients, and returns the
    first 20 CQCC coefficients per frame.

    Notes
    -----
    - The implementation uses `librosa.cqt` to compute the CQT with
      `hop_length=512` and `n_bins=84`.
    - The log conversion uses amplitude-to-dB on the complex CQT magnitude.
    - The DCT is applied along the frequency axis and the first 20 coefficients
      (low-order cepstral coefficients) are returned for each frame.

    Parameters
    ----------
    file_path : str
        Path to the input audio file.
    sr : int, optional
        Target sampling rate to resample the audio to when loading (default: 16000).

    Returns
    -------
    numpy.ndarray
        A 2-D array of shape (n_frames, 20) containing the first 20 CQCC
        coefficients for each frame. Returns None on failure.

    Raises
    ------
    RuntimeError
        If an underlying processing error occurs (file read error, missing
        dependency, numerical error), a RuntimeError will be raised with a
        descriptive message.
    """
    try:
        # Load audio (mono)
        y, sr_ret = librosa.load(file_path, sr=sr, mono=True)

        # Compute Constant-Q Transform (complex-valued matrix)
        # n_bins=84 and hop_length=512 as requested
        cqt = librosa.cqt(y, sr=sr_ret, hop_length=512, n_bins=84)

        # Convert magnitude to dB (log scale)
        magnitude = np.abs(cqt)
        # Use amplitude_to_db for stable log conversion
        log_cqt = librosa.amplitude_to_db(magnitude, ref=np.max)

        # Apply DCT along the frequency axis (axis=0) to get cepstral coefficients
        # Resulting shape: (n_bins, n_frames)
        cepstra = dct(log_cqt, type=2, axis=0, norm='ortho')

        # Take first 20 coefficients (low-order cepstral coefficients)
        n_coeff = 20
        if cepstra.shape[0] < n_coeff:
            # Zero-pad if frequency bins are fewer than requested coefficients
            pad_width = n_coeff - cepstra.shape[0]
            cepstra = np.pad(cepstra, ((0, pad_width), (0, 0)), mode='constant')

        cqcc = cepstra[:n_coeff, :].T  # shape -> (n_frames, n_coeff)

        return cqcc
    except RuntimeError as e:
        # Catch atexit errors specifically and suppress them
        if "can't register atexit after shutdown" in str(e):
            # Retry without the atexit issue
            import importlib
            importlib.reload(librosa)
            y, sr_ret = librosa.load(file_path, sr=sr, mono=True)
            cqt = librosa.cqt(y, sr=sr_ret, hop_length=512, n_bins=84)
            magnitude = np.abs(cqt)
            log_cqt = librosa.amplitude_to_db(magnitude, ref=np.max)
            cepstra = dct(log_cqt, type=2, axis=0, norm='ortho')
            n_coeff = 20
            if cepstra.shape[0] < n_coeff:
                pad_width = n_coeff - cepstra.shape[0]
                cepstra = np.pad(cepstra, ((0, pad_width), (0, 0)), mode='constant')
            cqcc = cepstra[:n_coeff, :].T
            return cqcc
        else:
            raise RuntimeError(f"Failed to extract CQCC features from {file_path}: {e}") from e
    except Exception as e:
        # Catch all other processing errors
        raise RuntimeError(f"Failed to extract CQCC features from {file_path}: {e}") from e


def extract_mel_spectrogram(
    file_path: str,
    sr: int = 16000,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> Optional[np.ndarray]:
    """Extract a normalized log-scaled Mel-spectrogram from an audio file.

    Parameters
    ----------
    file_path : str
        Path to the input audio file.
    sr : int
        Target sampling rate to resample the audio to (default: 16000).
    n_mels : int
        Number of Mel bands to generate (default: 128).
    n_fft : int
        FFT window size (default: 2048).
    hop_length : int
        Number of samples between successive frames (default: 512).

    Returns
    -------
    numpy.ndarray
        Normalized Mel-spectrogram (log scale) with values scaled to range [0, 1].
        Returns None on failure.

    Raises
    ------
    RuntimeError
        Raised when audio loading or processing fails.
    """
    try:
        # Load audio
        y, sr_ret = librosa.load(file_path, sr=sr, mono=True)

        # Compute mel-spectrogram (power)
        S = librosa.feature.melspectrogram(y=y, sr=sr_ret, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0)

        # Convert to log scale (dB)
        S_db = librosa.power_to_db(S, ref=np.max)

        # Min-max normalize to 0-1
        S_min = S_db.min()
        S_max = S_db.max()
        if S_max - S_min == 0:
            S_norm = np.zeros_like(S_db)
        else:
            S_norm = (S_db - S_min) / (S_max - S_min)

        return S_norm
    except Exception as e:
        raise RuntimeError(f"Failed to extract mel-spectrogram from {file_path}: {e}") from e
