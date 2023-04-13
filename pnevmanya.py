import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px

import platform

plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath


st.title("Pnevmanya kasalligini aniqlash")

#rasm yuklash
file = st.file_uploader("Rengen natijasini yuklang", type=['jpeg','png','gif','svg','jpg'])

if file:
    st.image(file)
    #PIL CONVERT
    img = PILImage.create(file)
    model = load_learner('Pnevmanya kllasifikatsiya.pkl')
    
    #predection
    pred, pred_id, probs = model.predict(img)
    st.success(f"Bashorat: {pred}")
    st.info(f"Ehtimolligi: {probs[pred_id]*100:.1f}%")

    #lotting
    fig = px.bar(y=probs*100, x=model.dls.vocab)
    st.plotly_chart(fig)
