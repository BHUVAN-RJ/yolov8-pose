from streamlit_webrtc import webrtc_streamer
import av
import streamlit as st
from ultralytics import YOLO
import cv2
import os
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException


model = YOLO('./best.pt')

def get_ice_servers():
    try:
        account_sid = os.environ["TWILIO_ACCOUNT_SID"]
        auth_token = os.environ["TWILIO_AUTH_TOKEN"]
    except KeyError:
        logger.warning(
            "Twilio credentials are not set. Fallback to a free STUN server from Google."  
        )
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

    client = Client(account_sid, auth_token)

    try:
        token = client.tokens.create()
    except TwilioRestException as e:
        st.warning(
            f"Error occurred while accessing Twilio API. Fallback to a free STUN server from Google. ({e}), {account_sid, auth_token}"
        )
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

    return token.ice_servers


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

    results = model(img)[0]

    for result in results:

        for outer_index, keypoints_batch in enumerate(result.keypoints.xy.tolist()):

            for inner_index, keypoint in enumerate(keypoints_batch):
                cv2.circle(img, (int(keypoint[0]), int(keypoint[1])), radius=3, color=(0, 0, 255), thickness=-1)

    return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(key="example", video_frame_callback=video_frame_callback, rtc_configuration={
        "iceServers": get_ice_servers(),
        "iceTransportPolicy": "relay",
    },media_stream_constraints={"video": True, "audio": False},
    async_processing=True)
