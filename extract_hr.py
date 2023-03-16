#execture with : conda activate pyvhr
# check example here : https://github.com/phuselab/pyVHR/blob/master/notebooks/pyVHR_run_on_video.ipynb

import os
from hr_analysis import analyse_folder

sources = "/home/pabloas/projects/open_source_AU/data_analysis/data/sharpened_db/*/*.mp4"

methods = ["HR-CNN", "MTTS-CAN", "cupy_POS", "cupy_CHROM", "cpu_LGI", "cpu_PBV", "cpu_GREEN", "cpu_OMIT", "cpu_ICA", "cpu_SSR"]
bpm_ests = ["median"]
roi_approachs = ["holistic"]

for roi_approach in roi_approachs:
        for bpm_est in bpm_ests:
                for method in methods:
                        target_folder = "analyses/"+ method + "_" + roi_approach + "_" + bpm_est+"/"
                        if os.path.exists(target_folder):
                                print("Skipping, already analysed : " + target_folder)
                                continue
                        print("Starting analysis " + target_folder)
                        os.mkdir(target_folder)
                        analyse_folder(sources, target_folder, bpm_est=bpm_est, method = method, roi_approach=roi_approach, wsize=1)