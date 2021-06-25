import streamlit as st

from stylegan2.utils import extract_audio, audio_features, timestamped_filename

# this caches the output to store the output and not call this function again
# and again preventing time wastage. `allow_output_mutation = True` tells the
# function to not hash the output of this function and we can get away with it
# because no arguments are passed through this.
# https://docs.streamlit.io/en/stable/api.html#streamlit.cache
@st.cache(allow_output_mutation=True, show_spinner=False)
def get_model():
    from stylegan2.model import StyleGAN2Model
    return {
        'StyleGAN2': StyleGAN2Model()
    }

# load all the models before the app starts
with st.spinner('Loading model...'):
    MODELS = get_model()

st.write('''
# StyleGAN2-Art
StyleGAN is a Generative Adversarial Network proposed by NVIDIA researchers.
It builds upon the Progressive Growing GAN architecture to produce photo-realistic \
synthetic images.

This model in particular is trained on trippy images to create cool \
artwork. The generated images are then synced with music of your choice!
''')

# select model
model_name = st.sidebar.selectbox(
    'Please select your app',
    ["StyleGAN2"]
)

if model_name != "StyleGAN2":
    st.write("Use `StyleGAN2` model!")
    model = MODELS['StyleGAN2']

if model_name == "StyleGAN2":
    st.write("### `StyleGAN2` Model")
    model = MODELS['StyleGAN2']

# upload audio file
st.write("Please upload an audio or a video file!")
uploaded_file = st.file_uploader("Audio/Video", type=['mp4', 'mp3', 'wav'])

if uploaded_file:
    # display audio player
    st.audio(uploaded_file)
    # extract and store audio file
    audio_path = extract_audio(uploaded_file)
    # get ratios for interpolation
    ratios = audio_features(file_path = audio_path)

# display video with audio synced with generated images
if st.button("Generate"):
    with st.spinner('Generating video...'):
        video_path = model.generate_interpolation(name = timestamped_filename(), ratios = ratios, sync_audio = True, audio_path = audio_path)
        video_file = open(video_path, 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)
