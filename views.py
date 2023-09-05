import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from scipy.stats import entropy


def draw_emotion_plot(scores, plot=None):
    df = pd.DataFrame(scores)
    mean = df.mean()
    df = pd.DataFrame({'emotion': mean.index, 'percentage': mean.values})
    fig = px.bar(df, x='emotion', y='percentage', color="emotion",
                 color_discrete_sequence=px.colors.qualitative.Set3)
    fig.update_yaxes(range=[0, 1])
    if plot is not None:
        plot.plotly_chart(fig)
    else:
        st.plotly_chart(fig)


class RealtimeView:
    def __init__(self):
        st.subheader('Facial Emotion')
        self.fer_plot = st.empty()

        st.subheader('Pace')
        self.speech_rate_session = st.empty()

        st.subheader('Speech Emotion')
        self.ser_plot = st.empty()

        st.subheader('Pitch')
        self.pitch_plot = st.empty()

        st.subheader('Script')
        self.text_output = st.empty()

    def update_view(self, stats):
        draw_emotion_plot(stats["fer_score"], self.fer_plot)
        draw_emotion_plot(stats["ser_score"], self.ser_plot)

        counts, bins = np.histogram(stats['pitch'], bins=range(200, 1000, 50))
        counts = counts / sum(counts)
        bins = 0.5 * (bins[:-1] + bins[1:])
        fig = px.bar(x=bins, y=counts, labels={'x': 'Pitch (Hz)', 'y': 'Density'})
        self.pitch_plot.plotly_chart(fig, use_container_width=True)

        if stats['wpm']:
            t, wpm = stats["wpm"][-1]
            with self.speech_rate_session.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    df = pd.DataFrame(stats['wpm'], columns=["time", "wpm"])
                    fig = px.line(df.iloc[1:], x="time", y="wpm", height=250)
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    st.metric(label="Speaking Rate", value=f"{wpm:.1f} wpm")
            self.text_output.caption(stats['text'])


def find_entropy(scores):
    df = pd.DataFrame(scores)
    mean = df.mean()
    pk = mean.values / mean.sum()
    # https://stats.stackexchange.com/questions/95261/why-am-i-getting-information-entropy-greater-than-1
    return entropy(pk, base=2)


class ReportView:
    def __init__(self, stats):
        col1, col2 = st.columns([0.37, 0.63])
        with col2:
            st.subheader("📋 Report")

        st.subheader("Metrics")
        duration, wpm = stats['wpm'][-1]

        # loudness may include -inf
        mean_loudness = np.mean(np.ma.masked_invalid(stats['loudness']))
        mean_pitch = np.mean(stats['pitch'])

        st.metric(label="Duration", value=f"{round(duration)} seconds")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Avg. Loudness", value=f"{mean_loudness:.1f} dB")
        with col2:
            st.metric(label="Avg. Pitch", value=f"{mean_pitch:.1f} Hz")
        with col3:
            st.metric(label="Avg. Speaking Rate", value=f"{wpm:.1f} wpm")

        if stats['fer_score']:
            fer_entropy = find_entropy(stats['fer_score'])
            num_face_emotions = len(stats['fer_score'][0])
        else:
            fer_entropy = 0
            num_face_emotions = 1

        if stats['ser_score']:
            ser_entropy = find_entropy(stats['ser_score'])
            num_speech_emotions = len(stats['ser_score'][0])
        else:
            ser_entropy = 0
            num_speech_emotions = 1

        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Face Emotion Diversity", value=f"{fer_entropy:.1f}")
        with col2:
            st.metric(label="Speech Emotion Diversity", value=f"{ser_entropy:.1f}")

        counts, bins = np.histogram(stats['pitch'], bins=range(200, 1000, 50))
        counts = counts / sum(counts)

        scale = 5
        st.subheader("Overall Performance")

        # the scores are mainly calculated by entropy / max_value of entropy
        df = pd.DataFrame(dict(
            theta=['Loudness', 'Pitch', 'Speaking Rate',
                   'Face Emotion', 'Speech Emotion'],
            r=[max(1, mean_loudness / 75) * scale,
               entropy(counts, base=2) / np.log2(len(counts)) * scale,
               max(1, wpm / 150) * scale,
               fer_entropy / np.log2(num_face_emotions) * scale,
               ser_entropy / np.log2(num_speech_emotions) * scale
               ]
        ))
        fig = px.line_polar(df, r='r', theta='theta', line_close=True)
        fig.update_traces(fill='toself')
        st.plotly_chart(fig, use_container_width=True)

        st.subheader('Script')
        st.caption(stats['text'])
        # st.json(stats)
