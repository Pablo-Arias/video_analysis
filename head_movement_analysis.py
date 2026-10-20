import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import glob
import time

import cv2
import mediapipe as mp
import numpy as np
import polars as pl

# MediaPipe face mesh indices. Left/right = the subject's.
RIGID = [10, 151, 9, 8, 168, 6, 197, 195, 5, 4,          # forehead midline, nose bridge, nose
         54, 284, 67, 297, 103, 332, 109, 338,           # forehead / temples
         127, 356, 139, 368,                             # temples near the ears
         33, 263, 133, 362]                              # eye corners (outer R, outer L, inner R, inner L)
LIDS = [159, 145, 158, 153, 160, 144, 386, 374, 385, 380, 387, 373]
IRIS = list(range(468, 478))                             # 468 = right iris centre, 473 = left iris centre
LMK_SUBSET = sorted(set(RIGID + LIDS + IRIS))

# eye geometry (subject's right eye = image left when not mirrored)
EYE = {"r": dict(outer=33, inner=133, upper=159, lower=145, iris=468, ear=(33, 160, 158, 133, 153, 144)),
       "l": dict(outer=263, inner=362, upper=386, lower=374, iris=473, ear=(362, 385, 387, 263, 373, 380))}


def euler_deg(R):
  """pitch (x, nod: > 0 = head down), yaw (y, turn: > 0 = nose towards image right), roll (z, tilt), in degrees."""
  pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
  yaw = np.degrees(np.arcsin(np.clip(-R[2, 0], -1, 1)))
  roll = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
  return pitch, yaw, roll


def eye_measures(P, e):
  """P: (478, 2) landmark pixels. Iris position in the eye's own frame, and eye aspect ratio (Soukupova & Cech 2016)."""
  o, i, u, d, c = P[e["outer"]], P[e["inner"]], P[e["upper"]], P[e["lower"]], P[e["iris"]]
  axis = o - i
  L = np.linalg.norm(axis)
  if L == 0:
    return np.nan, np.nan, np.nan
  ux = axis / L
  uy = np.array([-ux[1], ux[0]])
  x = np.dot(c - i, ux) / L
  # vertical: offset from the corner-to-corner line, in eye widths (+ = down in the image). Not scaled by the lid
  # opening, which collapses during blinks and narrows with smiles.
  y = np.dot(c - i, uy) / L * (1 if uy[1] >= 0 else -1)
  p1, p2, p3, p4, p5, p6 = (P[k] for k in e["ear"])
  ear = (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2 * np.linalg.norm(p1 - p4))
  return x, y, ear


def analyse_video(source
                  , target_analysis_folder
                  , target_processing_folder = "processing/"
                  , model_asset_path         = "face_landmarker_v2_with_blendshapes.task"
                  , export_lmks              = True
                  , max_frames               = None
                  ):
  """
  Head pose, iris gaze and eye aspect ratio for every frame of a video (MediaPipe Face Landmarker).

  Same frame loop and frame numbering as face_analysis_mp.analyse_video, so frame k here is frame k of the
  blendshape csv of the same video (frames with a repeated timestamp are skipped, as there).

  parameters:
      source                   : path to the video (.mp4)
      target_analysis_folder   : folder where the csv files will be saved
      target_processing_folder : folder where we keep track of the videos being analysed (see below)
      model_asset_path         : Default : "face_landmarker_v2_with_blendshapes.task". Path to the mediapipe model
      export_lmks              : Default : True. Also export <file_tag>_lmks.csv with a subset of landmarks
      max_frames               : Default : None. Stop after this many frames (for tests)

  outputs (in target_analysis_folder):
      <file_tag>.csv : one row per frame with a face:
          frame, pitch, yaw, roll (degrees, from the facial transformation matrix; pitch > 0 = head down,
          yaw > 0 = nose towards image right), tx, ty, tz (head position in cm, camera space; tz < 0, closer = larger),
          r00..r22 (rotation matrix), face_scale (inter-ocular distance in pixels),
          iris_x_l/r, iris_y_l/r (iris centre in its eye: x 0 = inner .. 1 = outer corner; y = offset below the
          corner line, in eye widths), ear_l, ear_r (eye aspect ratio), width, height
      <file_tag>_lmks.csv (if export_lmks) : long format (frame, idx, x, y, z) for the rigid points (forehead, nose
          bridge, temples, eye corners), the eyelids and the irises only. x, y are normalised image coordinates.

    Concerning the "processing" files
    Like face_analysis_mp, you can call this function several times on the HPC (e.g. one Slurm array task each):
    a video is claimed with a file in the "processing" folder, and videos already analysed are skipped.
    If a job is killed mid-file, delete that file's marker in the processing folder. Files are written under a
    temporary name and renamed at the end, so a killed job never leaves a partial csv behind.
  """
  file_tag = os.path.splitext(os.path.basename(source))[0]
  analysis_file = os.path.join(target_analysis_folder, file_tag + ".csv")
  lmk_file = os.path.join(target_analysis_folder, file_tag + "_lmks.csv")
  processing_file = os.path.join(target_processing_folder, file_tag)
  os.makedirs(target_analysis_folder, exist_ok=True)
  os.makedirs(target_processing_folder, exist_ok=True)

  #Skip if analysis exists or if someone else is processing it
  if os.path.isfile(analysis_file):
    print("Analysis already performed, skipping for : " + source)
    return
  try:
    os.close(os.open(processing_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY))   # atomic claim
  except FileExistsError:
    print("Analysis beeing done by someone else, skipping for : " + source)
    return

  print("Starting analysis for : " + source)
  t0 = time.perf_counter()
  options = mp.tasks.vision.FaceLandmarkerOptions(
      base_options=mp.tasks.BaseOptions(model_asset_path=model_asset_path),
      running_mode=mp.tasks.vision.RunningMode.VIDEO,
      output_face_blendshapes=False,
      output_facial_transformation_matrixes=True,
      num_faces=1)
  detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)
  cap = cv2.VideoCapture(source)
  W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  rows, lm_rows, tts, cpt = [], [], set(), 0
  while cap.isOpened():
    if not cap.grab():
      break
    flag, frame = cap.retrieve()
    tt = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    if tt in tts:                     # same rule as face_analysis_mp.analyse_video
      continue
    tts.add(tt)
    # face_analysis_mp feeds the decoded frame as is; kept identical so the detections match the blendshape run
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    res = detector.detect_for_video(image, timestamp_ms=tt)
    if res.face_landmarks:
      lm = res.face_landmarks[0]
      xyz = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float64)
      P = xyz[:, :2] * [W, H]
      M = np.array(res.facial_transformation_matrixes[0])
      R = M[:3, :3]
      pitch, yaw, roll = euler_deg(R)
      r = dict(frame=cpt, pitch=pitch, yaw=yaw, roll=roll, tx=M[0, 3], ty=M[1, 3], tz=M[2, 3],
               face_scale=np.linalg.norm(P[33] - P[263]))
      r.update({f"r{a}{b}": R[a, b] for a in range(3) for b in range(3)})
      for s, e in EYE.items():
        r[f"iris_x_{s}"], r[f"iris_y_{s}"], r[f"ear_{s}"] = eye_measures(P, e)
      rows.append(r)
      if export_lmks:
        lm_rows.append(np.column_stack([np.full(len(LMK_SUBSET), cpt), LMK_SUBSET, xyz[LMK_SUBSET]]))
    cpt += 1
    if max_frames and cpt >= max_frames:
      break
  cap.release()
  detector.close()
  print("Finished processing frames, processed " + str(cpt) + " frames, face in " + str(len(rows)) + ", for file : " + file_tag)

  if rows:
    df = pl.DataFrame(rows).with_columns(pl.exclude("frame").cast(pl.Float32).round(5),
                                         width=pl.lit(W), height=pl.lit(H))
    if export_lmks:
      L = np.vstack(lm_rows)
      lmk = pl.DataFrame({"frame": L[:, 0].astype(np.int64), "idx": L[:, 1].astype(np.int16),
                          "x": L[:, 2], "y": L[:, 3], "z": L[:, 4]}).with_columns(
          pl.col("x", "y", "z").cast(pl.Float32).round(5))
      lmk.write_csv(lmk_file + ".tmp")
      os.replace(lmk_file + ".tmp", lmk_file)
    # the main csv is written last: its presence means the video is done
    df.write_csv(analysis_file + ".tmp")
    os.replace(analysis_file + ".tmp", analysis_file)
  else:
    print("No face detected, nothing exported for file : " + file_tag)

  os.remove(processing_file)
  print("Finished all for : " + source + f" ({time.perf_counter() - t0:,.1f} s)")


## Analyse folder parallel
def analyse_video_parallel(sources
                  , target_analysis_folder
                  , target_processing_folder = "processing/"
                  , model_asset_path         = "face_landmarker_v2_with_blendshapes.task"
                  , export_lmks              = True
                  , max_frames               = None
                  , n_jobs                   = None
                  ):
  """
    sources should be in the shape of glob.glob (e.g. "preproc/prolific/*/trimed/*/*.mp4")
    Usage example:
    from head_movement_analysis import analyse_video_parallel
    def main():
        analyse_video_parallel("preproc/prolific/*/trimed/*/*-dry.mp4", target_analysis_folder="mp/head_movement/",
                               target_processing_folder="mp/head_movement_processing/",
                               model_asset_path="models/face_landmarker_v2_with_blendshapes.task")

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
                                        , repeat(model_asset_path)
                                        , repeat(export_lmks)
                                        , repeat(max_frames)
                                        ))


## ----------------------------------------------------------------------------------------------------------------
## Measures on the exported csv (validated nod measures: Hale et al. 2020, J Nonverbal Behav; Falk et al. 2023)
## ----------------------------------------------------------------------------------------------------------------

def load_head_movement(csv_path, n_frames=None):
  """Head movement csv on the full frame grid 0..n_frames-1 (frames without a face -> NaN). Returns a polars frame."""
  d = pl.read_csv(csv_path).unique(subset="frame", keep="first").with_columns(pl.col("frame").cast(pl.Int64))
  n = int(d["frame"].max()) + 1 if n_frames is None else n_frames
  grid = pl.DataFrame({"frame": np.arange(n, dtype=np.int64)})
  return grid.join(d, on="frame", how="left").sort("frame").with_columns(
      pl.col(pl.Float32).cast(pl.Float64)).with_columns(pl.col(pl.Float64).fill_null(float("nan")))


def _runs(mask):
  m = np.concatenate([[False], np.asarray(mask, bool), [False]])
  d = np.diff(m.astype(np.int8))
  return np.column_stack([np.flatnonzero(d == 1), np.flatnonzero(d == -1)])


def fill_short_gaps(x, max_gap=3):
  """Linear interpolation over runs of at most max_gap missing frames (never over longer gaps)."""
  x = np.asarray(x, float).copy()
  idx = np.arange(len(x))
  for s, e in _runs(~np.isfinite(x)):
    if s > 0 and e < len(x) and e - s <= max_gap:
      x[s:e] = np.interp(idx[s:e], [s - 1, e], [x[s - 1], x[e]])
  return x


def lowpass(x, fps, hz, order=2, min_len_s=1.0):
  """Zero-phase Butterworth low-pass on each continuous run of data."""
  from scipy.signal import butter, sosfiltfilt
  sos = butter(order, hz, btype="low", fs=fps, output="sos")
  out = np.full(len(x), np.nan)
  for s, e in _runs(np.isfinite(x)):
    if e - s >= min_len_s * fps:
      out[s:e] = sosfiltfilt(sos, x[s:e])
  return out


def head_angular_speed(df, fps=30, max_gap=3, lp_hz=8.0):
  """Head angular speed (deg/s) from pitch / yaw / roll low-passed at lp_hz (removes frame-level tracker jitter)."""
  a = [lowpass(fill_short_gaps(df[c].to_numpy(), max_gap), fps, lp_hz) for c in ("pitch", "yaw", "roll")]
  return np.sqrt(sum(np.gradient(x) ** 2 for x in a)) * fps


def morlet_cwt(x, fs, freqs, w0=6.0):
  """Continuous wavelet transform with a Morlet wavelet (Torrence & Compo 1998). Returns (W[n_freqs, n], scales)."""
  n = len(x)
  nfft = int(2 ** np.ceil(np.log2(2 * n)))
  X = np.fft.fft(x, nfft)
  omega = 2 * np.pi * np.fft.fftfreq(nfft, d=1 / fs)
  scales = (w0 + np.sqrt(2 + w0 ** 2)) / (4 * np.pi * np.asarray(freqs))
  W = np.empty((len(freqs), n), complex)
  for i, s in enumerate(scales):
    psi = np.pi ** -0.25 * np.exp(-0.5 * (s * omega - w0) ** 2) * (omega > 0) * np.sqrt(2 * np.pi * s * fs)
    W[i] = np.fft.ifft(X * np.conj(psi))[:n]
  return W, scales


def _fill(x):
  x = np.asarray(x, float)
  ok = np.isfinite(x)
  out = np.zeros_like(x)
  out[ok] = x[ok] - x[ok].mean()
  return out, ok


def coi_mask(ok, scales, fs):
  """True inside the cone of influence: within the Morlet e-folding time sqrt(2) * scale of the edges or of missing
  data (Torrence & Compo 1998; Grinsted et al. 2004). Hale et al. 2020 discard these points."""
  n = len(ok)
  idx = np.flatnonzero(np.concatenate([[True], ~np.asarray(ok, bool), [True]])) - 1
  t = np.arange(n)
  pos = np.searchsorted(idx, t)
  dist = np.minimum(np.abs(t - idx[np.clip(pos - 1, 0, len(idx) - 1)]), np.abs(idx[np.clip(pos, 0, len(idx) - 1)] - t))
  return dist[None, :] < (np.sqrt(2) * np.asarray(scales) * fs)[:, None]


def wavelet_power(x, fs, freqs, w0=6.0):
  """Wavelet power |W|^2 / fs [f, t] (units of x squared), NaN inside the cone of influence."""
  xf, ok = _fill(x)
  W, scales = morlet_cwt(xf, fs, freqs, w0)
  P = np.abs(W) ** 2 / fs
  P[coi_mask(ok, scales, fs)] = np.nan
  return P


def wavelet_coherence(x, y, fs, freqs, w0=6.0):
  """Wavelet coherence R2[f, t] in [0, 1] (Grinsted et al. 2004 smoothing: Gaussian in time, boxcar across scales),
  NaN inside the cone of influence."""
  from scipy.ndimage import gaussian_filter1d, uniform_filter1d
  xf, okx = _fill(x)
  yf, oky = _fill(y)
  Wx, scales = morlet_cwt(xf, fs, freqs, w0)
  Wy, _ = morlet_cwt(yf, fs, freqs, w0)
  inv = 1 / scales[:, None]

  def smooth(S):
    out = np.empty(S.shape, complex)
    for i, s in enumerate(scales):
      sd = max(s * fs, 1e-3)
      out[i] = gaussian_filter1d(S[i].real, sd, mode="constant") + 1j * gaussian_filter1d(S[i].imag, sd, mode="constant")
    k = max(int(round(0.6 / np.mean(np.diff(np.log2(scales))))), 1) if len(scales) > 1 else 1
    return uniform_filter1d(out.real, k, axis=0, mode="nearest") + 1j * uniform_filter1d(out.imag, k, axis=0, mode="nearest")

  Sxx = smooth(inv * np.abs(Wx) ** 2).real
  Syy = smooth(inv * np.abs(Wy) ** 2).real
  Sxy = smooth(inv * Wx * np.conj(Wy))
  with np.errstate(invalid="ignore", divide="ignore"):
    R2 = np.abs(Sxy) ** 2 / (Sxx * Syy)
  R2[coi_mask(okx & oky, scales, fs)] = np.nan
  return R2


def fast_nod_score(pitch, fps=30, band=(2.6, 6.5), mask=None, n_freqs=10):
  """Fast nod score (Falk et al. 2023, from Hale et al. 2020): mean Morlet wavelet power of head pitch in 2.6-6.5 Hz,
  outside the cone of influence, over the frames in `mask` (e.g. listening frames; default all)."""
  import warnings
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN columns inside the cone of influence
    P = np.nanmean(wavelet_power(fill_short_gaps(pitch), fps, np.geomspace(band[0], band[1], n_freqs)), axis=0)
  if mask is not None:
    P = np.where(mask, P, np.nan)
  return float(np.nanmean(P))


def slow_nod_coherence(pitch_a, pitch_b, fps=30, band=(0.2, 1.1), freqs=None):
  """Slow nod coherence (Hale et al. 2020): wavelet coherence of two partners' head pitch, averaged over 0.2-1.1 Hz
  and over time (cone of influence excluded). Compare real partners with pseudo partners (people who never met)."""
  freqs = np.geomspace(0.2, 8.0, 45) if freqs is None else freqs
  R2 = wavelet_coherence(fill_short_gaps(pitch_a), fill_short_gaps(pitch_b), fps, freqs)
  sel = (freqs >= band[0]) & (freqs <= band[1])
  return float(np.nanmean(R2[sel]))
