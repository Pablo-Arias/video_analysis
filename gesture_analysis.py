import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import glob
import time

import cv2
import mediapipe as mp
import numpy as np
import polars as pl

POSE_KEEP = list(range(25))                       # upper body; legs are never in view in webcam videos
SIDE = {"l": dict(shoulder=11, elbow=13, wrist=15), "r": dict(shoulder=12, elbow=14, wrist=16)}
THUMB_EVERY_S = 25                                 # one thumbnail every 25 s ...
THUMB_HANDS = 6                                    # ... plus up to 6 frames with hands, >= 10 s apart
MEA_W, MEA_ROWS, MEA_COLS, MEA_THR = 160, 6, 8, 10 # motion energy: width (px), grid, grey-level threshold


def draw(frame, hands, pose, label):
  """Annotated thumbnail: Pose points (green), hand points (red = 'Left', orange = 'Right'), a label."""
  img = frame.copy()
  H, W = img.shape[:2]
  if pose is not None:
    for k in POSE_KEEP:
      p = pose[k]
      if p.visibility > 0.5:
        cv2.circle(img, (int(p.x * W), int(p.y * H)), 4, (0, 200, 0), -1)
  for lms, hd in hands:
    col = (0, 0, 255) if hd.category_name == "Left" else (255, 128, 0)
    for p in lms:
      cv2.circle(img, (int(p.x * W), int(p.y * H)), 3, col, -1)
    w = lms[0]
    cv2.putText(img, f"{hd.category_name} {hd.score:.2f}", (int(w.x * W), int(w.y * H) + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
  cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
  return cv2.resize(img, (480, int(480 * H / W)))


def analyse_video(source
                  , target_analysis_folder
                  , target_processing_folder = "processing/"
                  , target_thumbs_folder     = "thumbs/"
                  , hand_model_asset_path    = "hand_landmarker.task"
                  , pose_model_asset_path    = "pose_landmarker_full.task"
                  , export_motion_energy     = True
                  , export_long_csvs         = True
                  , export_thumbnails        = False
                  , max_frames               = None
                  ):
  """
  Hands (MediaPipe Hand Landmarker), upper-body pose (Pose Landmarker) and motion energy for every frame of a video.

  Same frame loop and frame numbering as face_analysis_mp.analyse_video (frames with a repeated timestamp are skipped),
  so frame k here is frame k of the face csvs of the same video. Frames are converted BGR -> RGB for MediaPipe.

  parameters:
      source                   : path to the video (.mp4)
      target_analysis_folder   : folder where the csv files will be saved
      target_processing_folder : folder where we keep track of the videos being analysed (see below)
      target_thumbs_folder     : folder for annotated thumbnails (only with export_thumbnails)
      hand_model_asset_path    : Default : "hand_landmarker.task". Path to the mediapipe hand model
      pose_model_asset_path    : Default : "pose_landmarker_full.task". Path to the mediapipe pose model
      export_motion_energy     : Default : True. Export <file_tag>_mea.csv (motion energy, see below)
      export_long_csvs         : Default : True. Export <file_tag>_hands.csv and <file_tag>_pose.csv (all landmarks)
      export_thumbnails        : Default : False. Save annotated frames (every 25 s, plus up to 6 frames with hands)
                                 to check the detections visually
      max_frames               : Default : None. Stop after this many frames (for tests)

  outputs (in target_analysis_folder):
      <file_tag>.csv : wide, ONE ROW PER DECODED FRAME (NaN where nothing is detected):
          frame, n_hands, hand_score_max, pose (0/1); for the subject's left/right (l/r) side from Pose:
          shoulder_, elbow_, wrist_ {x, y, vis}; nose_x, nose_y; hand_{left,right}_{present,score,x,y} (hand centroid;
          MediaPipe handedness label as given, which assumes a mirrored image); width, height
      <file_tag>_hands.csv : long: frame, hand (label), hand_score, idx (0-20), x, y, z
      <file_tag>_pose.csv  : long: frame, idx (0-24, upper body), x, y, z, visibility, presence
      <file_tag>_mea.csv   : motion energy (Ramseyer & Tschacher 2011), one row per decoded frame: frame, me_all,
          me_r<row>c<col> on a 6 x 8 grid of the image = share of pixels whose grey level changed by > 10 since the
          previous frame (frame downscaled to 160 px wide, 3x3 blur). Regions (head / body) can be defined afterwards.
      x, y are normalised image coordinates.

  Note: with webcams that frame the face, the shoulders are often below the image; Pose then still reports them as
  visible but extrapolates their position. Check that the shoulders are inside the image (y < 1) before using them.

    Concerning the "processing" files
    As in face_analysis_mp, you can call this function several times on the HPC: a video is claimed with a file in
    the "processing" folder and videos already analysed are skipped. If a job is killed mid-file, delete that file's
    marker. Files are written under a temporary name and renamed at the end.
  """
  file_tag = os.path.splitext(os.path.basename(source))[0]
  out = lambda suffix: os.path.join(target_analysis_folder, file_tag + suffix + ".csv")  # noqa: E731
  processing_file = os.path.join(target_processing_folder, file_tag)
  os.makedirs(target_analysis_folder, exist_ok=True)
  os.makedirs(target_processing_folder, exist_ok=True)
  if export_thumbnails:
    os.makedirs(target_thumbs_folder, exist_ok=True)

  #Skip if analysis exists or if someone else is processing it
  if os.path.isfile(out("")):
    print("Analysis already performed, skipping for : " + source)
    return
  try:
    os.close(os.open(processing_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY))   # atomic claim
  except FileExistsError:
    print("Analysis beeing done by someone else, skipping for : " + source)
    return

  print("Starting analysis for : " + source)
  t0 = time.perf_counter()
  V = mp.tasks.vision
  hand = V.HandLandmarker.create_from_options(V.HandLandmarkerOptions(
      base_options=mp.tasks.BaseOptions(model_asset_path=hand_model_asset_path),
      running_mode=V.RunningMode.VIDEO, num_hands=2))
  pose = V.PoseLandmarker.create_from_options(V.PoseLandmarkerOptions(
      base_options=mp.tasks.BaseOptions(model_asset_path=pose_model_asset_path),
      running_mode=V.RunningMode.VIDEO, num_poses=1))
  cap = cv2.VideoCapture(source)
  W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps = cap.get(cv2.CAP_PROP_FPS) or 30
  rows, hand_rows, pose_rows, mea_rows, tts, cpt = [], [], [], [], set(), 0
  prev = None
  mh = int(round(MEA_W * H / W))
  ry = np.linspace(0, mh, MEA_ROWS + 1).astype(int)
  rx = np.linspace(0, MEA_W, MEA_COLS + 1).astype(int)
  last_hand_thumb, n_hand_thumbs = -10 ** 9, 0
  while cap.isOpened():
    if not cap.grab():
      break
    flag, frame = cap.retrieve()
    tt = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    if tt in tts:                     # same rule as face_analysis_mp.analyse_video
      continue
    tts.add(tt)
    if export_motion_energy:
      grey = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(frame, (MEA_W, mh), interpolation=cv2.INTER_AREA),
                                           cv2.COLOR_BGR2GRAY), (3, 3), 0)
      m = {"frame": cpt}
      if prev is not None:
        ch = (cv2.absdiff(grey, prev) > MEA_THR).astype(np.float32)
        m["me_all"] = float(ch.mean())
        for a in range(MEA_ROWS):
          for b in range(MEA_COLS):
            m[f"me_r{a}c{b}"] = float(ch[ry[a]:ry[a + 1], rx[b]:rx[b + 1]].mean())
      mea_rows.append(m)
      prev = grey
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    hr = hand.detect_for_video(image, timestamp_ms=tt)
    pr = pose.detect_for_video(image, timestamp_ms=tt)
    r = {"frame": cpt, "n_hands": len(hr.hand_landmarks),
         "hand_score_max": max((h[0].score for h in hr.handedness), default=np.nan)}
    hands = list(zip(hr.hand_landmarks, [h[0] for h in hr.handedness]))
    for side in ("left", "right"):
      r[f"hand_{side}_present"], r[f"hand_{side}_score"] = 0, np.nan
      r[f"hand_{side}_x"] = r[f"hand_{side}_y"] = np.nan
    for lms, hd in hands:
      side = hd.category_name.lower()
      xy = np.array([[p.x, p.y] for p in lms])
      if r[f"hand_{side}_present"] and hd.score <= r[f"hand_{side}_score"]:
        continue                      # two hands with the same label: keep the more confident one
      r[f"hand_{side}_present"], r[f"hand_{side}_score"] = 1, hd.score
      r[f"hand_{side}_x"], r[f"hand_{side}_y"] = xy[:, 0].mean(), xy[:, 1].mean()
      if export_long_csvs:
        for k, p in enumerate(lms):
          hand_rows.append((cpt, hd.category_name, hd.score, k, p.x, p.y, p.z))
    pl_ = pr.pose_landmarks[0] if pr.pose_landmarks else None
    r["pose"] = int(pl_ is not None)
    for s, idx in SIDE.items():
      for part, k in idx.items():
        p = pl_[k] if pl_ is not None else None
        r[f"{part}_{s}_x"] = p.x if p else np.nan
        r[f"{part}_{s}_y"] = p.y if p else np.nan
        r[f"{part}_{s}_vis"] = p.visibility if p else np.nan
    r["nose_x"], r["nose_y"] = (pl_[0].x, pl_[0].y) if pl_ is not None else (np.nan, np.nan)
    if pl_ is not None and export_long_csvs:
      for k in POSE_KEEP:
        p = pl_[k]
        pose_rows.append((cpt, k, p.x, p.y, p.z, p.visibility, p.presence))
    rows.append(r)
    if export_thumbnails:
      periodic = cpt % int(THUMB_EVERY_S * fps) == 0
      with_hand = bool(hands) and cpt - last_hand_thumb >= 10 * fps and n_hand_thumbs < THUMB_HANDS
      if periodic or with_hand:
        if with_hand and not periodic:
          last_hand_thumb, n_hand_thumbs = cpt, n_hand_thumbs + 1
        cv2.imwrite(os.path.join(target_thumbs_folder, f"{file_tag}_{cpt:05d}.jpg"),
                    draw(frame, hands, pl_, f"{file_tag[-40:]} f{cpt}"))
    cpt += 1
    if max_frames and cpt >= max_frames:
      break
  cap.release()
  hand.close()
  pose.close()
  print("Finished processing frames, processed " + str(cpt) + " frames, hands in "
        + str(sum(r['n_hands'] > 0 for r in rows)) + ", pose in " + str(sum(r['pose'] for r in rows))
        + ", for file : " + file_tag)

  if rows:
    tables = []
    if export_long_csvs:
      tables.append((pl.DataFrame(hand_rows, schema={"frame": pl.Int64, "hand": pl.Utf8, "hand_score": pl.Float32,
                                                     "idx": pl.Int16, "x": pl.Float32, "y": pl.Float32,
                                                     "z": pl.Float32}, orient="row")
                     .with_columns(pl.col("hand_score", "x", "y", "z").round(5)), "_hands"))
      tables.append((pl.DataFrame(pose_rows, schema={"frame": pl.Int64, "idx": pl.Int16, "x": pl.Float32,
                                                     "y": pl.Float32, "z": pl.Float32, "visibility": pl.Float32,
                                                     "presence": pl.Float32}, orient="row")
                     .with_columns(pl.col("x", "y", "z", "visibility", "presence").round(5)), "_pose"))
    if export_motion_energy:
      tables.append((pl.DataFrame(mea_rows, infer_schema_length=None)
                     .with_columns(pl.exclude("frame").cast(pl.Float32).round(5)), "_mea"))
    # the main csv is written last: its presence means the video is done
    tables.append((pl.DataFrame(rows).with_columns(pl.exclude("frame", "n_hands", "pose").cast(pl.Float32).round(5),
                                                   width=pl.lit(W), height=pl.lit(H)), ""))
    for d, suffix in tables:
      d.write_csv(out(suffix) + ".tmp")
      os.replace(out(suffix) + ".tmp", out(suffix))

  os.remove(processing_file)
  print("Finished all for : " + source + f" ({time.perf_counter() - t0:,.1f} s)")


## Analyse folder parallel
def analyse_video_parallel(sources
                  , target_analysis_folder
                  , target_processing_folder = "processing/"
                  , target_thumbs_folder     = "thumbs/"
                  , hand_model_asset_path    = "hand_landmarker.task"
                  , pose_model_asset_path    = "pose_landmarker_full.task"
                  , export_motion_energy     = True
                  , export_long_csvs         = True
                  , export_thumbnails        = False
                  , max_frames               = None
                  , n_jobs                   = None
                  ):
  """
    sources should be in the shape of glob.glob (e.g. "preproc/prolific/*/trimed/*/*-dry.mp4")
    Usage example:
    from gesture_analysis import analyse_video_parallel
    def main():
        analyse_video_parallel("preproc/prolific/*/trimed/*/*-dry.mp4", target_analysis_folder="mp/gesture/",
                               target_processing_folder="mp/gesture_processing/",
                               hand_model_asset_path="models/hand_landmarker.task",
                               pose_model_asset_path="models/pose_landmarker_full.task")

    if __name__ == '__main__':
        main()
    parameters: as analyse_video, plus
      n_jobs : Default : None (all cores). Number of parallel processes; set it to the cores you were given on the HPC.
  """
  os.makedirs(target_analysis_folder, exist_ok=True)
  os.makedirs(target_processing_folder, exist_ok=True)

  import multiprocessing
  from itertools import repeat
  pool_obj = multiprocessing.Pool(processes=n_jobs)
  sources = sorted(glob.glob(sources))

  pool_obj.starmap(analyse_video, zip(sources
                                        , repeat(target_analysis_folder)
                                        , repeat(target_processing_folder)
                                        , repeat(target_thumbs_folder)
                                        , repeat(hand_model_asset_path)
                                        , repeat(pose_model_asset_path)
                                        , repeat(export_motion_energy)
                                        , repeat(export_long_csvs)
                                        , repeat(export_thumbnails)
                                        , repeat(max_frames)
                                        ))
