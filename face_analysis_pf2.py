#Base distribution
import os
from os.path import exists
import glob
import sys

#External Packages
from feat import Detectorv1

def analyse_videos_feat2(sources, target_folder
						, batch_size=100
						, output_size=700
						, face_model = "retinaface"
						, landmark_model = "mobilefacenet"
						, au_model = "xgb"
						, emotion_model = "resmasknet"
						, device = "cuda"
						, task_id = None
						, n_tasks = None
						):
	"""
		Same as analyse_videos (face_analysis_pf.py) but with py-feat 2.x (Detectorv1, the modular py-feat pipeline).
		Analyse the source videos and save one csv per video in target folder.
		Sources follows the sytax of glob.glob

		sources = "data/sharpened_db/*/*.mp4"
		target_folder = "au_analysis/"
		analyse_videos_feat2(sources, target_folder)

		output_size : frames are rescaled to this size, as py-feat 0.6 detect_video did by default (700)
		identity and gaze models are not run (not needed for AUs and emotions)
		task_id, n_tasks : to split the videos between several jobs (e.g. a slurm array) : this job analyses every n_tasks-th video, starting at task_id

		Requires py-feat >= 2.0 (Detectorv1). Check detector parameters here : https://py-feat.org
	"""

	#New detector
	detector = Detectorv1(
	    face_model       = face_model,
	    landmark_model   = landmark_model,
	    au_model         = au_model,
	    emotion_model    = emotion_model,
	    identity_model   = None,
	    gaze_model       = None,
	    device           = device
	)
	os.makedirs(target_folder, exist_ok=True)

	#Videos analysed by this job
	files = sorted(glob.glob(sources))
	if task_id is not None:
		files = files[task_id::n_tasks]

	for file in files:
		file_tag = os.path.splitext(os.path.basename(file))[0]
		target_file = target_folder + file_tag + ".csv"
		try:
			if not exists(target_file):
				#Empty file first, so that other jobs skip this video
				open(target_file, "a").close()
				print("Analysing " + file + " ....", flush=True)
				video_prediction = detector.detect(file
												, data_type = "video"
												, output_size = output_size
												, batch_size = batch_size
												, progress_bar = False
												)
				video_prediction.to_csv(target_file)

			else:
				print(file + " analysis exists, skipping it", flush=True)

		except KeyboardInterrupt:
			print("Keyboard interrupt")
			os.remove(target_file)
			sys.exit(0)

		except Exception as e:
			print("An error occured analysing : " + file)
			print(e)
			os.remove(target_file)
			pass
