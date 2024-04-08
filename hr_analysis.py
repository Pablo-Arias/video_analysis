#Base distribution
import os
from os.path import exists
import glob
import pickle
import sys

#External Packages
from pyVHR.analysis.pipeline import Pipeline
from pyVHR.plot.visualize import *
from pyVHR.utils.errors import getErrors, printErrors, displayErrors

from pyVHR.analysis.pipeline import DeepPipeline

#Local Scripts
from conversions import get_file_without_path

def analyse_folder(sources, target_folder, wsize = 6, roi_approach = 'patches'
				   , bpm_est = 'clustering', method = 'cpu_CHROM'
				   , pre_filt = True
				   , post_filt= True
				   , cuda=True
				   , verb=True
				   ):
	"""
		Analyse the source and save datatrame results in target folder
		Sources has 
		source = "data/sharpened_db/*/*.mp4"
		target_folder = "au_analysis/"

		analyse_videos(sources, target_folder)

		# params
		wsize = 6                  # window size in frames
		roi_approach = 'patches'   # 'holistic' or 'patches'
		bpm_est = 'clustering'     # BPM final estimate, if patches choose 'medians' or 'clustering'
		method = 'cpu_CHROM'       # one of the methods implemented in pyVHR

		BEWARE for deep methods not all parameters work!!!

		For deep methods, check available parameters here : 
			https://github.com/phuselab/pyVHR/blob/55cbdf9efc51977f9670520b7362e0f386de7e4b/pyVHR/analysis/pipeline.py#L811

	"""

	for file in glob.glob(sources):
		file_tag = get_file_without_path(file)
		target_file = target_folder + file_tag + ".pickle"

		if exists(target_file):
			print(target_file + ' exists, skipping it')
			continue
		#try:
			# run
		if method in ["HR_CNN", "MTTS_CAN"]:
			pipe = DeepPipeline()          # object to execute the pipeline
			res = pipe.run_on_video(file,
										method=method,
										post_filt=post_filt,
										cuda=cuda, 
										verb=verb
										)			
		else:
			pipe = Pipeline()          # object to execute the pipeline
			res = pipe.run_on_video(file,
										winsize=wsize, 
										roi_method='convexhull',
										roi_approach=roi_approach,
										method=method,
										estimate=bpm_est,
										patch_size=0, 
										RGB_LOW_HIGH_TH=(5,230),
										Skin_LOW_HIGH_TH=(5,230),
										pre_filt=pre_filt,
										post_filt=post_filt,
										cuda=cuda, 
										verb=verb
										)

				# ERRORS
				#RMSE, MAE, MAX, PCC, CCC, SNR = getErrors(bvps, fps, bpmES, bpmGT, timesES, timesGT)
				#printErrors(RMSE, MAE, MAX, PCC, CCC, SNR)
				#displayErrors(bpmES, bpmGT, timesES, timesGT)

				## file
				#print(bvps)
				#print(timeES)
				#print(bpmES)
				#res = [bvps, timeES, bpmES]

			with open(target_file, 'wb') as handle:
				pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)

		# except KeyboardInterrupt:
		# 	print("interrupted")
		# 	sys.exit(0)
		# except Exception as e:
		# 	print("An error occured analsing : " + file)
		# 	print(e)
		# 	pass


if __name__ == "__main__":
	sources = "/home/pabloas/projects/open_source_AU/data_analysis/data/sharpened_db/*/*.mp4"
	target_folder = "aanalysis/"
	analyse_videos(sources, target_folder)
