import logging.handlers
import threading
import time
from collections import deque
from pathlib import Path
from typing import List

from views import RealtimeView, ReportView

import queue
from fer import FER
from huggingface_hub import hf_hub_download
import cv2
from stt import Model
from speech_emotion_recognition import SpeechEmotionRecognition
from pitch_detection import find_pitch
import av
import numpy as np
import pydub
import streamlit as st
import datetime

from streamlit_webrtc import WebRtcMode, webrtc_streamer

from PIL import Image, ImageFont
from pilmoji import Pilmoji

np.seterr(divide='ignore', invalid='ignore')

HERE = Path(__file__).parent

logger = logging.getLogger(__name__)


@st.cache_resource
def get_fer_model():
    return FER()


@st.cache_resource
def get_stt_model():
    REPO_ID = "mbarnig/lb-de-fr-en-pt-coqui-stt-models"
    en_stt_model_path = hf_hub_download(repo_id=REPO_ID, filename="english/model.tflite")
    en_stt_scorer_path = hf_hub_download(repo_id=REPO_ID, filename="english/huge-vocabulary.scorer")

    model = Model(en_stt_model_path)
    model.enableExternalScorer(en_stt_scorer_path)
    return model


@st.cache_resource
def get_ser_model():
    return SpeechEmotionRecognition()


@st.cache_resource
def emoji_font():
    font = ImageFont.truetype(str(HERE / "resources/NotoColorEmoji-Regular.ttf"), size=40)
    return font


EMOJI_FONT = emoji_font()


def draw_emoji(img, emoji, xy):
    img_pil = Image.fromarray(img)
    pilmoji = Pilmoji(img_pil)
    pilmoji.text(xy, emoji, (0, 0, 0), align="center", font=EMOJI_FONT)
    return np.asarray(img_pil)


def main():
    st.header("Presentation Trainer")
    st.markdown("""This demonstration application combines speech recognition, 
    speech emotion recognition, and facial emotion recognition 
    to assist presenters in practicing their presentations.""")

    frames_deque_lock = threading.Lock()
    frames_deque: deque = deque([])

    stats = {
        "wpm": [(0, 0)],
        "pitch": [],
        "ser_score": [],
        "fer_score": [],
        "loudness": [0]
    }

    async def queued_audio_frames_callback(frames: List[av.AudioFrame]) -> List[av.AudioFrame]:
        with frames_deque_lock:
            frames_deque.extend(frames)
        # Return empty frames to be silent.
        new_frames = []
        for f in frames:
            input_array = f.to_ndarray()
            new_frame = av.AudioFrame.from_ndarray(
                np.zeros(input_array.shape, dtype=input_array.dtype),
                layout=f.layout.name,
            )
            new_frame.sample_rate = f.sample_rate
            new_frames.append(new_frame)
        return new_frames

    fer_model = get_fer_model()
    video_result_queue = queue.Queue()  # TODO: A general-purpose shared state object may be more useful.

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        detections = fer_model.detect_emotions(img)

        color = (255, 255, 255)
        for detection in detections:
            box = detection["box"]
            (startX, startY, width, height) = box
            endX = startX + width
            endY = startY + height

            highlighted_emotion = sorted(detection["emotions"].items(), key=lambda x: -x[1])[0][0]
            confidence = detection["emotions"][highlighted_emotion]

            # display the prediction
            label = f"{highlighted_emotion}: {round(confidence * 100, 2)}%"
            cv2.rectangle(img, (startX, startY), (endX, endY), color, thickness=2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(
                img,
                label,
                (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                thickness=2
            )

        # NOTE: This `recv` method is called in another thread,
        # so it must be thread-safe.
        for detection in detections:
            # save the prediction
            video_result_queue.put(detection["emotions"])

        # !!! pillow is using RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # display on the top
        last_wpm = stats.get("wpm", [0, 70])[-1][1]
        emoji_list = []

        if last_wpm < 50:
            emoji_list.append("🐌")
        elif last_wpm > 130:
            emoji_list.append("😵‍💫")

        if stats.get('loudness', [0])[-1] < 70:
            emoji_list.append("📢")

        if not emoji_list:
            emoji_list.append("😀")

        img = draw_emoji(img, "".join(emoji_list), (int(img.shape[1] * 0.48), 6))

        return av.VideoFrame.from_ndarray(img, format="rgb24")

    # called after starting or stopping the streamer
    def on_change():
        st.session_state['stats'] = stats

    webrtc_ctx = webrtc_streamer(
        key="presentation-trainer",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        queued_audio_frames_callback=queued_audio_frames_callback,
        # run locally using None, or remote with a STUN server
        rtc_configuration=None,  # {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": True},
        on_change=on_change
    )

    page = None
    if webrtc_ctx.state.playing:
        page = RealtimeView()
    else:
        if 'stats' not in st.session_state or len(st.session_state['stats']["loudness"]) == 1:
            return
        page = ReportView(st.session_state['stats'])
        return

    stats['start_time'] = datetime.datetime.now()
    # for STT
    stt_model = get_stt_model()
    stream = stt_model.createStream()

    # for SER
    ser_model = get_ser_model()

    # initialize sound buffer
    sound_window_len = 3000  # 3s
    sound_window_buffer = pydub.AudioSegment.silent(
        duration=sound_window_len
    )
    while webrtc_ctx.state.playing:
        # --------------------------- VIDEO ---------------------------
        while not video_result_queue.empty():
            try:
                pred = video_result_queue.get(block=False)
                stats["fer_score"].append(pred)
            except queue.Empty:
                break

        # --------------------------- AUDIO ---------------------------
        # flush audio frames
        audio_frames = []
        with frames_deque_lock:
            while len(frames_deque) > 0:
                frame = frames_deque.popleft()
                audio_frames.append(frame)

        if len(audio_frames) == 0:
            time.sleep(0.1)
            continue

        sound_chunk = pydub.AudioSegment.empty()
        for audio_frame in audio_frames:
            sound_chunk += pydub.AudioSegment(
                data=audio_frame.to_ndarray().tobytes(),
                sample_width=audio_frame.format.bytes,
                frame_rate=audio_frame.sample_rate,
                channels=len(audio_frame.layout.channels),
            )

        if len(sound_chunk) > 0:
            sound_window_buffer += sound_chunk
            # get the latest 3s buffer
            if len(sound_window_buffer) > sound_window_len:
                sound_window_buffer = sound_window_buffer[-sound_window_len:]
            sound_window_buffer = sound_window_buffer.set_channels(1)  # Stereo to mono
            # dB SPL
            loudness = 20 * np.log10(10 ** (sound_window_buffer.dBFS / 20) / 20e-6)
            stats['loudness'].append(loudness)
            # -inf means no sound

            # SER
            if loudness >= 65:
                pred = ser_model.inference(sound_window_buffer)
                stats["ser_score"].append(pred)

            # Pitch
            stats["pitch"].append(find_pitch(sound_window_buffer))

            # STT (not using sound buffer)
            sound_chunk = sound_chunk.set_channels(1).set_frame_rate(
                stt_model.sampleRate()
            )
            buffer = np.array(sound_chunk.get_array_of_samples())
            stream.feedAudioContent(buffer)
            # text = stream.intermediateDecode()
            # text_output.markdown(f"**Script:** {text}")
            transcript = stream.intermediateDecodeWithMetadata().transcripts[0]

            if transcript.tokens:
                text = "".join(t.text for t in transcript.tokens)
            else:
                text = ""
            stats['text'] = text
            stats['last_update'] = datetime.datetime.now()
            stats['duration'] = (stats['last_update'] - stats['start_time']).total_seconds()

            if stats['duration'] > 1:
                words = text.split(" ")
                current_wpm = len(words) / stats['duration'] * 60
                stats["wpm"].append((stats['duration'], current_wpm))

            page.update_view(stats)


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
               "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.ERROR)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.ERROR)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)

    main()
