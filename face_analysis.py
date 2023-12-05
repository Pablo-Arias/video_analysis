#Base distribution
import os
from os.path import exists
import glob
import sys

#External Packages
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from feat import Detector
from feat.plotting import draw_facepose, draw_lineface
import numpy as np
from torchvision.io import read_video
from feat.utils.io import read_feat

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
	    face_model       = face_model,
	    landmark_model   = landmark_model,
	    au_model         = au_model,
	    emotion_model    = emotion_model,
	    facepose_model   = facepose_model,
	    device           = device
	)
	try:
		os.mkdir(target_folder)
	except:
		pass

	for file in glob.glob(sources):
		try:
			file_tag = get_file_without_path(file)
			target_file = target_folder + file_tag + ".csv"
			if not exists(target_file):
				open(target_file, "a")
				print("Analysing "+  file + " ....")
				video_prediction = detector.detect_video(file
		                                         , skip_frames = skip_frames
		                                         , batch_size = batch_size
		                                         , num_workers = num_workers
		                                         , pin_memory = pin_memory
		                                         , n_jobs = n_jobs
												 )

				video_prediction.to_csv(target_file)
			
			else:
				print(file + 'analysis exists, skipping it')			

		except KeyboardInterrupt:
			print("Keyboard interrupt")
			os.remove(target_file)
			sys.exit(0)

		except Exception as e:
			print("An error occured analysing : " + file)
			print(e)
			os.remove(target_file)
			pass

def extract_au_analysis_frames(analysis, target_folder,faceboxes=False, add_titles=False, muscles=True, plot_original_image=False, gazes=True):
	#Create analysis folder
	os.makedirs(target_folder, exist_ok=True)

	#Import analysis
	video_prediction = read_feat(analysis)

	#Get plots
	figures = video_prediction.loc[:].plot_detections(faceboxes=faceboxes, add_titles=add_titles, muscles=muscles, plot_original_image=plot_original_image, gazes=gazes)

	#Save plots
	for cpt, figure in enumerate(figures):
		figure.savefig(target_folder+str(cpt)+".png")


def extract_tracked_frames(analysis
                            , target_folder, facelines =True,  pose=True, face_detection=True, landmarks=True
                            , lmk_color = "w"
                            , face_detect_color="cyan"
                            , lmk_lw = 1
                            , face_detection_lw = 1
                            , my_dpi = 10
                            ):

	#Create analysis folder
	os.makedirs(target_folder, exist_ok=True)


	#Load video
	video_prediction = read_feat(analysis)
	file = video_prediction.input[0]
	video, audio, info = read_video(file, output_format="TCHW")

	frames = video_prediction.frame.values

	for frame_nb in frames:
		#Get image from video 
		img = video[frame_nb, :, :]

		#Plot original image
		fig, face_ax = plt.subplots(frameon=False)
		face_ax.imshow(img.permute([1, 2, 0]))

		#Plot face detection
		if face_detection:
			frame = video_prediction.loc[video_prediction["frame"] == frame_nb]
			facebox = frame[video_prediction.facebox_columns].values[0]
			rect = Rectangle(
				(facebox[0], facebox[1]),
				facebox[2],
				facebox[3],
				linewidth=face_detection_lw,
				edgecolor=face_detect_color,
				fill=False,
				)
			face_ax.add_patch(rect)

		#Pose
		if pose:
			face_ax = draw_facepose(
								pose=frame[video_prediction.facepose_columns].values[0],
								facebox=facebox,
								ax=face_ax,
								)

		#Landmarks
		if landmarks:
			landmark = frame[frame.landmark_columns].values[0]
			currx = landmark[:68]
			curry = landmark[68:]

		# facelines
		if facelines:
			face_ax = draw_lineface(
					currx
					, curry
					, ax=face_ax
					, color=lmk_color
					, linewidth=lmk_lw
					)

		plt.axis('off')
		plt.savefig(target_folder + str(frame_nb)+".png", bbox_inches='tight', dpi=200, pad_inches = 0)
		plt.close()

def create_tracked_video(analysis, target_video_folder="preproc/tracked/", target_frames_folder= "preproc/frames/", fps=30, img_extension=".png", preset="slow", remove_frames=False, extract_frames=True, create_video=True, video_extension=".mp4", add_audio=None):
	"""
	This function takes the results from py-feat and creates a video with them showing the tracking, gaze and head detection results.
	"""
	from face_analysis import extract_tracked_frames
	from video_processing import create_movie_from_frames
	import os
	from conversions import get_file_without_path
	import shutil

	#define variables
	file_tag = get_file_without_path(analysis)
	target_video = target_video_folder + file_tag + video_extension

	#Extract frames
	if extract_frames:
		os.makedirs(target_frames_folder, exist_ok=True)
		extract_tracked_frames(analysis, target_frames_folder)

	#Extract audio
	if add_audio:
		video_prediction = read_feat(analysis)
		file = video_prediction.input[0]
		from video_processing import extract_audio, replace_audio
		import uuid
		audio_file = str(uuid.uuid1()) +"____.wav"
		extract_audio(file, target_name=audio_file, nb_audio_channels=1)
	else:
		audio_file=None
	
	# Create video from frames
	if create_video:
		os.makedirs(target_video_folder, exist_ok=True)
		create_movie_from_frames(frame_name_tag=target_frames_folder, fps=fps, img_extension =img_extension , target_video=target_video, preset=preset, audio_file=audio_file)

	if add_audio:
		os.remove(audio_file)

	if remove_frames:
		shutil.rmtree(target_frames_folder)

def create_au_video(analysis, target_video_folder="preproc/tracked/"
							, target_frames_folder= "preproc/frames/"
							, fps=30, img_extension=".png"
							, preset="slow"
							, remove_frames=False
							, extract_frames=True
							, create_video=True
							, video_extension=".mp4"
							, add_audio=None ):
	"""
	Creates a video of the AUs extracted with py-feat by extracting each frame and then collecting it all into a video file with ffmpeg
	
	"""							
	from face_analysis import extract_au_analysis_frames
	from video_processing import create_movie_from_frames
	import os
	from conversions import get_file_without_path
	import shutil

	#Extract frames
	if extract_frames:
		os.makedirs(target_frames_folder, exist_ok=True)
		extract_au_analysis_frames(analysis, target_frames_folder)
	
	#Extract audio
	if add_audio:
		video_prediction = read_feat(analysis)
		file = video_prediction.input[0]
		from video_processing import extract_audio, replace_audio
		import uuid
		audio_file = str(uuid.uuid1()) +"____.wav"
		extract_audio(file, target_name=audio_file, nb_audio_channels=1)	
	else:
		audio_file = None

	# Create video from frames
	if create_video:
		file_tag = get_file_without_path(analysis)
		os.makedirs(target_video_folder, exist_ok=True)
		create_movie_from_frames(frame_name_tag=target_frames_folder, fps=fps, img_extension =img_extension , target_video=target_video_folder + file_tag + video_extension, preset=preset, audio_file=audio_file)

	if remove_frames:
		shutil.rmtree(target_frames_folder)

if __name__ == "__main__":
	sources = "data/sharpened_db/*/*.mp4"
	target_folder = "au_analysis/"

	#analyse_videos(sources, target_folder)

