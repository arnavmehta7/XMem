from pathlib import Path
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2, os
from utils import *
from PIL import Image
import numpy as np

stroke_width = st.sidebar.slider("Stroke width", 1, 30, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex", value='#9B002C')

# 0 for upload
# 1 for draw & submit
# 2 to show tracked video

def img_video(dir, output_name):
    imgs = os.listdir(dir)
    imgs = sorted(imgs, key=lambda x: int(x.split("_")[1].split(".")[0]))
    imgs = [os.path.join(dir, ig) for ig in imgs]
    print(' All files: ', imgs)
    frame_size = (640, 480)
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
    clips = ImageSequenceClip(imgs, 25)
    clips.write_videofile(output_name)

def save_mask(dir:str, mask: Image):
    color_map_np = np.frombuffer(pal, dtype=np.uint8).reshape(-1, 3).copy()
    color_map_np = (color_map_np.astype(np.float32)*1.5).clip(0, 255).astype(np.uint8)
    assert isinstance(mask, np.ndarray)
    colored_mask = color_map_np[mask]
    Image.fromarray(colored_mask).save(dir)

# get only the first frame of video and show it as a image if video is uploaded
def get_first_frame(video):
    save_input_dir = os.path.join('results', Path(video.name).stem)
    create_dir(save_input_dir)
    st.session_state['base_dir'] = save_input_dir
    print('Directory: ', save_input_dir)
    save_input_path = os.path.join(save_input_dir, 'video.mp4')
    with open(save_input_path, 'wb') as video_file:
        print('Saved Input Video at: ', save_input_path)
        video_file.write(video.getbuffer())
    cap = cv2.VideoCapture(save_input_path)
    ret, frame = cap.read()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


if 'step' not in st.session_state:
    st.session_state['step'] = 0
# upload the video to run the model on
if st.session_state['step'] == 0:
    video = st.file_uploader("Upload a video", type=["mp4"])
    if video is not None: 
        st.session_state['video'] = True
        st.session_state['image'] = get_first_frame(video)
        # st.image(st.session_state['image'])
        st.session_state['step'] = 1

if st.session_state['step'] == 1:
    canvas_result = st_canvas(
        fill_color="#9B002C",  # Fixed fill color with some opacity
        stroke_color=stroke_color,
        stroke_width=stroke_width,
        background_image=Image.fromarray(st.session_state['image']),
        update_streamlit=True,
        drawing_mode="freedraw",
        display_toolbar=True,
        key="canvas")
    # button to submit the canvas
    if canvas_result.image_data is not None:
        btn = st.button('Submit')
        if btn:
            # save image
            img = canvas_result.image_data.copy()
            img[np.where(img[:, :, 3] == 0)] = (0, 0, 0, 255)
            cv2.imwrite(os.path.join(st.session_state['base_dir'], 'canvas.png'), cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            # save_mask(os.path.join(st.session_state['base_dir'], 'canvas.png'), np.argmax(img, axis=0).astype(np.uint8))
            
            mask = cv2.imread(os.path.join(st.session_state['base_dir'], 'canvas.png'))
            # colored_mask =  Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY).astype(np.uint8)).convert('P')
            colored_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            colored_mask = (colored_mask>0).astype(np.uint8)
            colored_mask = Image.fromarray(colored_mask).convert('P')
            colored_mask.putpalette(pal)
            print('session state image shape ', st.session_state['image'].shape[:2][::-1])
            colored_mask = colored_mask.resize(st.session_state['image'].shape[:2][::-1]) 
            colored_mask.save(os.path.join(st.session_state['base_dir'], 'init_mask.png'))
            del colored_mask, mask, img
            st.session_state['step'] = 2
            
if st.session_state['step'] == 2:
    st.info('Video is being tracked')
    with st.spinner():
        st.write('Quality of the output is reduced to reduce resource consumptions')
        if 'masking' not in st.session_state:
            cap = cv2.VideoCapture(os.path.join(st.session_state['base_dir'], 'video.mp4'))
            plain_mask = np.array(Image.open(os.path.join(st.session_state['base_dir'], 'init_mask.png')))
            print('plain mask shape: ', plain_mask.shape)
            get_mask_from_video_stream(cap, st.session_state['base_dir'], plain_mask)
        
        tracked_obj_video = os.path.join(st.session_state['base_dir'], 'tracked_obj.mp4')
        img_video(os.path.join(st.session_state['base_dir'], 'overlay'), tracked_obj_video)
        
        if os.path.exists(tracked_obj_video):
            st.success('Video is tracked')
            st.session_state['masking'] = True
            st.video(open(tracked_obj_video,'rb').read())
