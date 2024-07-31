
This tutorial runs you through step by step instructions of how to use the video analysis scripts to extract face features.

#Conda environment
First creaate your environment
```
conda create --name mediapipe python=3.11
conda activate mediapipe
pip install mediapipe scipy soundfile polars
conda develop PATH_TO_STIM_REPO
conda develop PATH_TO_video_analysis_repo
```

Note that you have to change here PATH_TO_STIM_REPO (https://github.com/Pablo-Arias/STIM) and the PATH_TO_video_analysis_repo (https://github.com/Pablo-Arias/video_analysis) which should be the path to the repositories. Some people have issues with conda develop, if you do, you also more simply just append the path to your system at the begining of your script (import sys; sys.path.append(path_to_repo).

# Download mediapipe model
Then download the model for data analysis and put it inside a folder called models:
!wget -O face_landmarker_v2_with_blendshapes.task -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

# Prepare data
Then make sure that your folder has all the preprocessed videos (i.e. that the folders “preproc/mkcalsoup_test313/trimed/*/*.mp4” exist). If not, change the sources in the script below so it has access to all .mp4 files.

# Run the code:

Then try to run this code. Make sure to change the sources variable so that it points towards your preprocessed videos.

```
from face_analysis_mp import analyse_video_parallel, analyse_video
import glob
import os

model = "models/face_landmarker_v2_with_blendshapes.task"

#One file at a time
sources = "preproc/mkcalsoup_test313/trimed/*/*.mp4"

for file in glob.glob(sources):
    print("Processing file : " + file)
    analyse_video(file
                        , target_analysis_folder   = "mp_sd/au_analysis/"
                        , target_frames_folder     = "mp_sd/tracked/"
                        , target_video_folder      = "mp_sd/tracked_video/"
                        , target_au_video_folder   = "mp_sd/au_video/"
                        , target_AU_plots_folder   = "mp_sd/AU_bar_graph_folder/"
                        , combined_videos_folder   = "mp_sd/combined_videos_folder/"
                        , target_processing_folder = "mp_sd/processing/" 
                        , model_asset_path         = model
                        , export_tracked_frames    = True
                        , delete_frames            = True
                        , delete_bar_graphs        = True
                        , export_analysis          = True
                        , export_AU_bargraphs      = True
                        , create_tracked_video     = True
                        , combine_AU_graphs_into_video = True
                        , combine_AU_bargraphs_and_tracked_video = True
                        )
```