import os
from typing import List
import supervision as sv
import cv2
import pandas as pd
import torch
from PIL import Image
import open_clip
import glob
import time
import subprocess
import base64
import datetime

class UserError(Exception):
    pass


NO_GOOD_IMAGES_FOUND = "ERROR: No images found in the 'good' directory ( {} ).\n"
NO_BAD_IMAGES_FOUND = "ERROR: No images found in the 'bad' directory ( {} ).\n"


class LabeledEmbeddings(object):
    def __init__(self, embeddings: torch.Tensor, labels: List[bool]):
        self.embeddings = embeddings
        self.labels = labels

class ModelConfig(object):
    def __init__(self, model_name: str = 'hf-hub:imageomics/bioclip', pretrained: str = None):
        self.model_name = model_name
        self.pretrained = pretrained

class Argus(object):
    def __init__(self, project_dir: str, model_config: ModelConfig):
        self.project_paths = ProjectPaths(project_dir)
        model, _, preprocess = open_clip.create_model_and_transforms(model_config.model_name,
                                                                     pretrained=model_config.pretrained)
        self.model = model
        self.preprocess = preprocess

    def create_labeled_embeddings(self) -> LabeledEmbeddings:
        labels = []
        embeddings = []
        good_paths = self.project_paths.get_good_image_paths()
        bad_paths = self.project_paths.get_bad_image_paths()
        if not good_paths:
            raise UserError(NO_GOOD_IMAGES_FOUND.format(self.project_paths.good_dir))
        if not bad_paths:
            raise UserError(NO_BAD_IMAGES_FOUND.format(self.project_paths.bad_dir))
        for path in good_paths:
            embeddings.append(self.create_image_embedding_for_path(path))
            labels.append(True)
        for path in bad_paths:
            embeddings.append(self.create_image_embedding_for_path(path))
            labels.append(False)
        return LabeledEmbeddings(torch.cat(embeddings, dim=0), labels)

    def create_image_embedding_for_path(self, path) -> torch.Tensor:
        img = cv2.imread(path)
        return self.create_image_embedding(img)

    def create_image_embedding(self, img) -> torch.Tensor:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img)
        image = self.preprocess(image).unsqueeze(0)
        image_features = self.model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features


class ProjectPaths(object):
    def __init__(self, project_dir: str):
        self.project_dir = project_dir
        self.frames_dir = project_dir
        self.good_dir = os.path.join(self.frames_dir, 'good')
        self.bad_dir = os.path.join(self.frames_dir, 'bad')
        self.bad_dir = os.path.join(self.frames_dir, 'bad')
        self.tmp_dir = os.path.join(self.frames_dir, 'tmp')

    def create_dirs(self):
        os.makedirs(self.project_dir, exist_ok=True)
        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.good_dir, exist_ok=True)
        os.makedirs(self.bad_dir, exist_ok=True)
        os.makedirs(self.tmp_dir, exist_ok=True)

    def get_good_image_paths(self) -> str:
        return glob.glob(os.path.join(self.good_dir, '*.png'))

    def get_bad_image_paths(self) -> str:
        return glob.glob(os.path.join(self.bad_dir, '*.png'))

    def create_predictions_path(self) -> str:
        temp_time_str = time.strftime('%Y%m%d%H%M%S', time.gmtime())
        return os.path.join(self.project_dir, f"predictions_{temp_time_str}.csv")

    def create_clips_path(self) -> str:
        temp_time_str = time.strftime('%Y%m%d%H%M%S', time.gmtime())
        return os.path.join(self.project_dir, f"clips_{temp_time_str}.csv")


def create_png_data_url_img(frame, maxsize = (256,256)):
    img = cv2.resize(frame,maxsize,interpolation=cv2.INTER_AREA)
    is_success, buffer = cv2.imencode(".png", img)
    if is_success:
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        return f"<img src='data:image/png;base64,{encoded_image}'>"
    else:
        return "Error converting frame to PNG"


class PredictionReport(object):
    def __init__(self, sample_dir: str, frame_wait_seconds: float):
        self.sample_dir = sample_dir
        self.frame_wait_seconds = frame_wait_seconds
        self.content = ""

    def add_video_path_header(self, video_path: str):
        self.content += f"<h2>{video_path}</h2>"

    def add_image(self, frame, prob_str, label, time_str):
        img_tag = create_png_data_url_img(frame)
        self.content += f"""<div>
            {img_tag} Label: {label} Prob: {prob_str} Time: {time_str}
        </div>"""

    def __str__(self) -> str:
        return f"""<html><body>
            <h1>Prediction Report using {self.sample_dir} frame_wait_seconds: {self.frame_wait_seconds}</h1>
            {self.content}
        </body></html>"""


class ClipsReport(object):
    def __init__(self):
        pass

    def __str__(self) -> str:
        return f"""<html><body>
            <h1>Clips Report</h1>
            TODO: Show images before clip begins, when the clip starts, when clip ends, and after clip ends.
        </body></html>"""


def extract_sample_images(sample_dir: str, video_path: str, seconds_interval: float = 5.0, max_samples = None):
    project_paths = ProjectPaths(sample_dir)
    project_paths.create_dirs()
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    stride = round(fps * seconds_interval)

    print("Extracting frames from", video_path, "to", sample_dir)
    cnt = 0
    with sv.ImageSink(target_dir_path=project_paths.frames_dir) as sink:
        frames_generator = sv.get_video_frames_generator(source_path=video_path, stride=stride)
        for idx, frame in enumerate(frames_generator):
            filename = f"{idx}.png"
            sink.save_image(image=frame, image_name=filename)
            cnt += 1
            if max_samples:
                if cnt >= max_samples:
                    print(f"Limit of {max_samples} reached.")
                    break
    print(f"Frames extracted to {project_paths.frames_dir}.\n")
    print(f"""Directory structure:

 {sample_dir}
 ├── good/
 └── bad/

Add images of target frames to the "good" directory ( {project_paths.good_dir} )
and unwanted frames to the "bad" directory ( {project_paths.bad_dir} ).""")


def predict_frames(sample_dir: str, video_paths: List[str], model_config: ModelConfig,
                   frame_wait_seconds: float = 1.0,
                   report_path: str = None, report_interval: int = 5, save_images: bool = False,):
    report = PredictionReport(sample_dir, frame_wait_seconds)
    argus = Argus(sample_dir, model_config)
    print("Creating image embeddings for good and bad images...")
    labeled_embeddings = argus.create_labeled_embeddings()

    movie_paths_ary = []
    probs_ary = []
    label_index_ary = []
    label_ary = []
    time_str_ary = []

    with torch.no_grad(), torch.cuda.amp.autocast():
        for video_path in video_paths:
            report.add_video_path_header(video_path)
            print("Finding clips in", video_path)
            video = cv2.VideoCapture(video_path)
            fps = video.get(cv2.CAP_PROP_FPS)
            stride = round(fps * frame_wait_seconds) # every 1 second
            cnt = 0
            with sv.ImageSink(target_dir_path=argus.project_paths.tmp_dir) as sink:
                frames_generator = sv.get_video_frames_generator(source_path=video_path, stride=stride)
                for idx, frame in enumerate(frames_generator):
                    image_embedding = argus.create_image_embedding(frame)
                    probs = (100.0 * image_embedding @ labeled_embeddings.embeddings.T).softmax(dim=-1)
                    movie_paths_ary.append(video_path)
                    prob_str = f"{probs.max().item():.2f}"
                    probs_ary.append(prob_str)
                    label_idx = probs.argmax().item()
                    label_index_ary.append(label_idx)
                    label_ary.append(labeled_embeddings.labels[label_idx])
                    seconds = (idx * stride) / fps
                    td = datetime.timedelta(seconds=seconds)
                    time_str = str(td)
                    time_str_ary.append(time_str)
                    if save_images:
                        sink.save_image(image=frame, image_name=f"{idx}.png")
                    if cnt % report_interval == 0:
                        report.add_image(frame, prob_str, labeled_embeddings.labels[label_idx], time_str)
                    cnt += 1

    df = pd.DataFrame({
        "video_file": movie_paths_ary,
        "probs": probs_ary,
        "label_index": label_index_ary,
        "label": label_ary,
        "time": time_str_ary
    })
    
    if report_path:
        with open(report_path, "w") as outfile:
            outfile.write(str(report))
    return df


def find_clips(predictions: pd.DataFrame, report_path: str = None):
    df = pd.DataFrame(predictions)

    # group by consecutive values
    df['consec'] = df.label.ne(df.label.shift()).cumsum()
    # filter out frames that weren't matched
    df = df[df.label == True]
    grouped_df = df.groupby('consec').agg({'time': ['min', 'max'], 'video_file': ['first']})
    grouped_df.columns = grouped_df.columns.get_level_values(1)
    grouped_df.rename(columns={'min': 'start', 'max': 'end', 'first': 'video_file'}, inplace=True)
    print(f"Clips found:")
    for _, row in grouped_df.iterrows():
        print(row['video_file'], row['start'], row['end'])
    print("")

    if report_path:
        with open(report_path, "w") as outfile:
            outfile.write(f"<html><body><h1>TODO</h1>")
    return grouped_df


def extract_clips(clips_df: pd.DataFrame, clips_dir: str):
    print("Extracting clips")
    os.makedirs(clips_dir, exist_ok=True)
    output_paths = []
    for idx, row in clips_df.iterrows():
        video_file = row['video_file']
        video_file_prefix = os.path.splitext(os.path.basename(video_file))[0]
        start = row['start']
        end = row['end']
        outpath = os.path.join(clips_dir, f"{video_file_prefix}_{start}.mp4").replace(":", "-")
        if os.path.exists(outpath):
            print("Removing existing clip", outpath)
            os.unlink(outpath)
        print(f"Running ffmpeg to extract clip {idx} to {outpath}.")
        subprocess.check_output(f"ffmpeg -hide_banner -loglevel error -i '{video_file}' -ss '{start}' -to '{end}' {outpath}", shell=True)
        output_paths.append(outpath)

    print("\n\nExtracted clips paths:")
    for output_path in output_paths:
        print(output_path)
    print("")
    return output_paths