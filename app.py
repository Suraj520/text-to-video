import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

"""
# Welcome to VAA's Demo - Text to Video !
"""

import streamlit as st
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

# Function to generate video and return its path
def generate_video(prompt):
    device = torch.device('cpu')
    pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w", torch_dtype=torch.float16, device=device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    
    video_frames = pipe(prompt, num_inference_steps=40, height=320, width=576, num_frames=24).frames
    video_path = export_to_video(video_frames[0])
    
    return video_path

# Streamlit app
def main():
    st.title('Video Generation with Diffusers')
    
    # Input prompt
    prompt = st.text_input('Enter prompt', 'Two people talking about weather in a car')
    
    # Button to generate video
    if st.button('Generate Video'):
        st.write('Generating video...')
        video_path = generate_video(prompt)
        st.write('Video generated!')
        
        # Display video
        st.video(video_path, format='video/mp4')
        
        # Download link for the generated video
        st.markdown(f"### [Download Video](/{video_path})")

if __name__ == '__main__':
    main()
