from tkinter.ttk import Style
from turtle import color
from typing import List

import cv2
import torch
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.colors as mcolors
from PIL import Image
from streamlit_webrtc import *
import av
import time

CLASSES = ['Casque_NO','Casque_OK','Gilet_NO', 'Gilet_OK']
# 0 pour Casque_NO
# 1 pour Casque_OK
# 2 pour Gilet_NO
# 3 pour Gilet_OK

WEBRTC_CLIENT_SETTINGS = ClientSettings(
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )


st.set_page_config(
    page_title="Uniforme detection",
)

st.title("Détection de l'uniforme")

# Functions
# --------------------------------------------

@st.cache(max_entries=2)
def get_yolo5():
    return torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/sylvia.penfeunteun/Documents/cas_pratique/model/new_models/best.pt', force_reload=False)

@st.cache(max_entries=10)
def get_preds(img : np.ndarray) -> np.ndarray:
    return model([img]).xyxy[0].numpy()

def get_colors(indexes : List[int]) -> dict:
    to_255 = lambda c: int(c*255)
    tab_colors = list(mcolors.TABLEAU_COLORS.values())
    tab_colors = [list(map(to_255, mcolors.to_rgb(name_color))) 
                                                for name_color in tab_colors]
    base_colors = list(mcolors.BASE_COLORS.values())
    base_colors = [list(map(to_255, name_color)) for name_color in base_colors]
    rgb_colors = tab_colors + base_colors
    rgb_colors = rgb_colors*5

    color_dict = {}
    for i, index in enumerate(indexes):
        if i < len(rgb_colors):
            color_dict[index] = rgb_colors[i]
        else:
            color_dict[index] = (255,0,0)

    return color_dict

def get_legend_color(class_name : int):
    index = CLASSES.index(class_name)
    color = rgb_colors[index]
    return 'background-color: rgb({color[0]},{color[1]},{color[2]})'.format(color=color)

class VideoProcessor(VideoProcessorBase):
    
    def __init__(self):
        self.model = model
        self.rgb_colors = rgb_colors
        self.target_class_ids = target_class_ids
        self.lablab = False

    def get_preds(self, img : np.ndarray) -> np.ndarray:
        return self.model([img]).xyxy[0].numpy()

    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
        lab = []
        result = self.get_preds(frm)
        result = result[np.isin(result[:,-1], self.target_class_ids)]
        
        for bbox_data in result:
            xmin, ymin, xmax, ymax, _, label = bbox_data
            p0, p1, label = (int(xmin), int(ymin)), (int(xmax), int(ymax)), int(label)
            frm = cv2.rectangle(frm, 
                                    p0, p1, 
                                    self.rgb_colors[label], 2) 
        if label not in lab:
            lab.append(label)
            self.lablab = lab

        #return cv2.cvtColor(frm, cv2.COLOR_RGB2BGR)
        return av.VideoFrame.from_ndarray(frm, format='bgr24')

# présence de l'uniforme

uniforme_val = st.sidebar.empty()
message = "UNIFORME NON VERIFIE"
uniforme_val.markdown(f'<p style="color:red">{message}</p>',  unsafe_allow_html=True)

# zone de chargement du model

with st.spinner('Loading the model...'):
    model = get_yolo5()




# UI elements
# ----------------------------------------------------

# barre du côté gauche
    
all_labels_chbox = st.sidebar.checkbox('Liste de classes', value=True)


# section de prédiction

# cible des labels et leurs couleurs

if all_labels_chbox:
    target_class_ids = list(range(len(CLASSES)))
else:
    target_class_ids = [0]

rgb_colors = get_colors(target_class_ids)
detected_ids = None


ctx = webrtc_streamer(
    key="example", 
    video_processor_factory= VideoProcessor,
    #video_html_attrs=VideoHTMLAttributes( autoPlay=True, controls=True, style={"width": "100%"}))
    client_settings=WEBRTC_CLIENT_SETTINGS,)


if ctx.video_processor:
    ctx.video_processor.model = model
    ctx.video_processor.rgb_colors = rgb_colors
    ctx.video_processor.target_class_ids = target_class_ids


detected_ids = set(detected_ids if detected_ids is not None else target_class_ids)
labels = [CLASSES[index] for index in detected_ids]
legend_df = pd.DataFrame({'label': labels})
st.sidebar.dataframe(legend_df.style.applymap(get_legend_color))


while True:
    time.sleep(1)
    if ctx.video_processor.lablab != False:
        lablab = ctx.video_processor.lablab
        print(lablab)
        print("size:", len(lablab))
        if (0 not in lablab and 2 not in lablab) and (1 in lablab and 3 in lablab): # 0: Casque_NO, 1: Casque_OK, 2: Gilet_NO, 3: Gilet_OK
            message = "UNIFORME VERIFIE"
            uniforme_val.markdown(f'<p style="color:green">{message}</p>',  unsafe_allow_html=True)
        else:
            message = "UNIFORME NON VERIFIE"
            uniforme_val.markdown(f'<p style="color:red">{message}</p>',  unsafe_allow_html=True)
