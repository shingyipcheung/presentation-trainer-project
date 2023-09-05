from utils import audiosegment_to_librosawav
import pydub
import statsmodels.api as sm
from scipy.signal import find_peaks


# https://scicoding.com/pitchdetection/
def find_pitch(sound_window_buffer: pydub.AudioSegment):
    y = audiosegment_to_librosawav(sound_window_buffer)
    # autocorrelation
    auto = sm.tsa.acf(y, nlags=2000)
    peaks = find_peaks(auto)[0]  # Find peaks of the autocorrelation
    lag = peaks[0]  # Choose the first peak as our pitch component lag
    pitch = sound_window_buffer.frame_rate / lag  # Transform lag into frequency
    return pitch
