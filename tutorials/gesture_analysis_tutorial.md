This tutorial runs you through the steps to extract hands, upper-body pose and motion energy from videos with `gesture_analysis.py`.

# Conda environment
Same environment as for the face analysis (see `face_analysis_tutorial.md`):
```
conda create --name mediapipe python=3.11
conda activate mediapipe
pip install mediapipe scipy polars
conda develop PATH_TO_video_analysis_repo
```

# Download the mediapipe models
Put them in a folder called models:
```
wget -O models/hand_landmarker.task -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
wget -O models/pose_landmarker_full.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task
```

# Run the code (1 video at a time)
```
from gesture_analysis import analyse_video
import glob

for file in glob.glob("preproc/prolific/*/trimed/*/*-dry.mp4"):
    analyse_video(file
                  , target_analysis_folder   = "mp/gesture/"
                  , target_processing_folder = "mp/gesture_processing/"
                  , target_thumbs_folder     = "mp/gesture_thumbs/"
                  , hand_model_asset_path    = "models/hand_landmarker.task"
                  , pose_model_asset_path    = "models/pose_landmarker_full.task"
                  , export_motion_energy     = True
                  , export_long_csvs         = True    # all hand / pose landmarks (larger files)
                  , export_thumbnails        = False   # True: annotated frames to check the detections
                  )
```
Outputs per video (frame numbering as `face_analysis_mp`, so the csvs line up with the face analysis):
- `<file_tag>.csv`: one row per frame, with hand presence and centroids, shoulders / elbows / wrists / nose from Pose;
- `<file_tag>_hands.csv` and `<file_tag>_pose.csv`: all landmarks, long format;
- `<file_tag>_mea.csv`: motion energy (Ramseyer & Tschacher 2011), the share of changed pixels per cell of a 6 x 8 grid.

See the docstring of `analyse_video` for every column.

**Check the framing first.** Webcams usually frame the face. In the DuckSoup videos, the shoulders were below the image in ~20 % of recordings, and hands were visible in a median 5.8 % of frames. Pose still reports out-of-frame shoulders as visible, but extrapolates them. Before using a body measure, keep only frames where the landmarks are inside the image (y < 1).

# Run the code (in parallel)
```
from gesture_analysis import analyse_video_parallel

if __name__ == '__main__':
    analyse_video_parallel(sources = "preproc/prolific/*/trimed/*/*-dry.mp4"
                           , target_analysis_folder   = "mp/gesture/"
                           , target_processing_folder = "mp/gesture_processing/"
                           , hand_model_asset_path    = "models/hand_landmarker.task"
                           , pose_model_asset_path    = "models/pose_landmarker_full.task"
                           , n_jobs = 16)
```

# On the HPC (Slurm array)
Each task calls `analyse_video` on the whole list. The tasks share the videos through the processing folder, and finished videos are skipped.
```
#!/bin/bash -l
#SBATCH --job-name=gesture_analysis
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#SBATCH --array=0-15
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0-06:00:00
module load apps/miniforge
source activate mediapipe
python -u -c "
import glob, gesture_analysis as ga
for f in sorted(glob.glob('preproc/prolific/*/trimed/*/*-dry.mp4')):
    ga.analyse_video(f, 'mp/gesture/', 'mp/gesture_processing/', hand_model_asset_path='models/hand_landmarker.task',
                     pose_model_asset_path='models/pose_landmarker_full.task')"
```
If a task is killed mid-file, delete that file's marker in the processing folder and resubmit.
The same pattern works for `head_movement_analysis.py` (see `head_movement_tutorial.ipynb`).
