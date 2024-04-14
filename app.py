from streamlit_webrtc import webrtc_streamer
import av
from ultralytics import YOLO
import cv2
import os
from twilio.rest import Client

model = YOLO('best.pt')
account_sid = os.environ['TWILIO_ACCOUNT_SID']
auth_token = os.environ['TWILIO_AUTH_TOKEN']
client = Client(account_sid, auth_token)

token = client.tokens.create()

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    results = model(img)[0]

    for result in results:

        for outer_index, keypoints_batch in enumerate(result.keypoints.xy.tolist()):

            for inner_index, keypoint in enumerate(keypoints_batch):
                cv2.circle(img, (int(keypoint[0]), int(keypoint[1])), radius=3, color=(0, 0, 255), thickness=-1)

    return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(key="example", video_frame_callback=video_frame_callback, rtc_configuration={
        "iceServers": token.ice_servers,
        "iceTransportPolicy": "relay",
    },media_stream_constraints={"video": True, "audio": False},
    async_processing=True)
