import streamlit as st
from replicate.client import Client

replicate = Client(api_token="r8_bGbJmY2Z5OwhYL5ohi111dgupcARGQT0zAb3y")
"""
# Welcome to VAA's Demo - Text to Video !
"""
# Streamlit app
def main():
    
    # Input prompts
    prompt = st.text_area('Enter prompt', 'Two people talking about weather in a car')
    
    # Button to generate video
    if st.button('Generate Video'):
        st.write('Generating video...')
        input_data = {
            "fps": 24,
            "width": 1024,
            "height": 576,
            "prompt": prompt,
            "guidance_scale": 10.5,
        }
        
        # Generate video using replicate.run
        try:
            output_data = replicate.run(
                "anotherjesse/zeroscope-v2-xl:9f747673945c62801b13b84701c783929c0ee784e4748ec062204894dda1a351",
                input=input_data
            )
            video_url = output_data.get('output_video')
            st.write('Video generated!')
            
            # Display download button for the generated video
            st.markdown(f"### [Download Video]({video_url})")
        except Exception as e:
            st.write(f'Failed to generate video: {e}')

if __name__ == '__main__':
    main()
