import fire
from argus.core import init_project, extract_images, predict_frames, find_clips, extract_clips, UserError

# 1 make a folder and put all your videos in there
# 2 within that folder run `argus save-frames <video.mp4>`
# 3 move the frames into the good and bad folders
# 4 run `argus extract-clips`

def todo():
    print("Figure this out")

def main():
    try:
        fire.Fire({
            'todo': todo,
        }, name='argus')
    except UserError as e:
        print(e)
