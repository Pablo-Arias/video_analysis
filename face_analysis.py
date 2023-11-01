#Base distribution
import os
from os.path import exists
import glob
import sys

#External Packages
import pandas as pd
import matplotlib.pyplot as plt
from feat import Detector

#Local Scripts
from conversions import get_file_without_path

def analyse_videos(sources, target_folder
						, skip_frames=1
						, batch_size=900
						, num_workers=16
						, pin_memory=False
						, n_jobs = 12
						, face_model = "retinaface"
						, landmark_model = "mobilefacenet"
						, au_model = 'xgb'
						, emotion_model = "resmasknet"
						, facepose_model = "img2pose"
						, device = "cuda"
						):
	"""
		Analyse the source and save datatrame results in target folder
		Sources follows the sytax of glob.glob
		sources = "data/sharpened_db/*/*.mp4"
		target_folder = "au_analysis/"

		analyse_videos(sources, target_folder)

		Check detector parameters here : https://py-feat.org/pages/api.html

	"""
	
	#New detector
	detector = Detector(
	    face_model = face_model,
	    landmark_model = landmark_model,
	    au_model = au_model,
	    emotion_model = emotion_model,
	    facepose_model = facepose_model,
	    device = device
	)

	for file in glob.glob(sources):
		try:
			file_tag = get_file_without_path(file)
			target_file = target_folder + file_tag + ".csv"
			if not exists(target_file):
				print("Analysing "+  file + " ....")
				video_prediction = detector.detect_video(file
		                                         , skip_frames = skip_frames
		                                         , batch_size = batch_size
		                                         , num_workers = num_workers
		                                         , pin_memory = pin_memory
		                                         , n_jobs = n_jobs
												, face_model = "retinaface"
												, landmark_model = "mobilefacenet"
												, au_model = 'xgb'
												, emotion_model = "resmasknet"
												, facepose_model = "img2pose"
												, device = "cuda"												 
												 )

				video_prediction.to_csv(target_file)
			else:
				print(file + ' exists, skipping it')

		except KeyboardInterrupt:
			print("Keyboard interrupt")
			sys.exit(0)

		except Exception as e:
			print("An error occured analysing : " + file)
			print(e)
			pass


if __name__ == "__main__":
	sources = "data/sharpened_db/*/*.mp4"
	target_folder = "au_analysis/"

	#analyse_videos(sources, target_folder)
