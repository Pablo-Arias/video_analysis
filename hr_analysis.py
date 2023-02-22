#Base distribution
import os
from os.path import exists
import glob
import pickle

#External Packages
from pyVHR.analysis.pipeline import Pipeline
from pyVHR.plot.visualize import *
from pyVHR.utils.errors import getErrors, printErrors, displayErrors


#Local Scripts
from conversions import get_file_without_path

def analyse_folder(sources, target_folder, wsize = 6, roi_approach = 'patches', bpm_est = 'clustering', method = 'cpu_CHROM'):
	"""
		Analyse the source and save datatrame results in target folder
		Sources has 
		source = "data/sharpened_db/*/*.mp4"
		target_folder = "au_analysis/"

		analyse_videos(sources, target_folder)

		# params
		wsize = 6                  # window size in seconds
		roi_approach = 'patches'   # 'holistic' or 'patches'
		bpm_est = 'clustering'     # BPM final estimate, if patches choose 'medians' or 'clustering'
		method = 'cpu_CHROM'       # one of the methods implemented in pyVHR

	"""

	for file in glob.glob(sources):
		wsize = 8 # seconds of video processed (with overlapping) for each estimate 
		file_tag = get_file_without_path(file)
		target_file = target_folder + file_tag + ".pickle"

		if exists(target_file):
			print(target_file + ' exists, skipping it')
			continue
		try:
			# run
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
				                                        pre_filt=True,
				                                        post_filt=True,
				                                        cuda=True, 
				                                        verb=True)

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
		except
			print("An error occured analsing : " + file)
		
		except KeyboardInterrupt:
			print("interrupted")
			sys.exit(0)


if __name__ == "__main__":
	sources = "/home/pabloas/projects/open_source_AU/data_analysis/data/sharpened_db/*/*.mp4"
	target_folder = "aanalysis/"
	analyse_videos(sources, target_folder)
