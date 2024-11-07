import os
os.environ['OPENBLAS_NUM_THREADS'] = '8'

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
import cv2
from matplotlib import pyplot as plt
import ntpath
from video_processing import create_movie_from_frames, extract_audio, combine_2_videos
import os
import shutil
import polars as pl
import glob


#We implemented some functions to visualize the face landmark detection results. <br/> Run the following cell to activate the functions.
def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image

def plot_face_blendshapes_bar_graph(face_blendshapes):
  # Extract the face blendshapes category names and scores.
  face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
  face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
  # The blendshapes are ordered in decreasing score value.
  face_blendshapes_ranks = range(len(face_blendshapes_names))

  fig, ax = plt.subplots(figsize=(12, 12))
  bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
  ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
  ax.invert_yaxis()

  # Label each bar with values
  for score, patch in zip(face_blendshapes_scores, bar.patches):
    plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

  ax.set_xlabel('Score')
  ax.set_title("Face Blendshapes")
  ax.set_xlim(0, 1)  # Set X-axis limit from 0 to 1

  plt.tight_layout()
  return fig

def analyse_video(source
                  , target_analysis_folder
                  , target_frames_folder     = "tracked/"
                  , target_video_folder      = "output_video/"
                  , target_au_video_folder   = "au_video/"
                  , target_AU_plots_folder   = "AU_bar_graph_folder/"
                  , combined_videos_folder   = "combined_videos_folder/"
                  , target_processing_folder = "processing/"
                  , model_asset_path         = "face_landmarker_v2_with_blendshapes.task"
                  , output_face_blendshapes  = True
                  , num_faces                = 1
                  , export_tracked_frames    = True
                  , export_AU_bargraphs      = True
                  , export_blendshapes       = True
                  , export_lmks              = False
                  , create_tracked_video     = True
                  , combine_AU_graphs_into_video = True
                  , combine_AU_bargraphs_and_tracked_video = True
                  , fps            = 25
                  , delete_frames  = True
                  , delete_bar_graphs = True
                  ):
  """
  parameters:
      target_analysis_folder : The name of the csv folder where the csv file with the face analysis will be saved
      target_frames_folder  : The name of the folder where the frames will be saved (to create a video, we need to create different frames, with e.g. face tracking results)
      target_video_folder    : The name of the folder where the video with the tracking will be created.
      target_au_video_folder   : The name of the folder where the video that recreates faces from AUs will be created
      target_AU_plots_folder   : The name of the folder where the plots to create the video with AUs will be created
      combined_videos_folder  : The name of the folder where the combined videos (with tracking and AUs will be created)
      target_processing_folder :  The name of the folder where we will keep track f the videos being analysed at each time.
      model_asset_path            : Default : “face_landmarker_v2_with_blendshapes.task”. Name of the the mediapipe model to use
      output_face_blendshapes  : Default : True; If the model should compute blendhapes (e.g. smiling , frowning , etc...)
      num_faces                = 1; Number of faces to detect in the video
      export_tracked_frames    = True, If the script should export tracked frames.Tracked frames are visual representations of the faces with the landmarks within it..
      export_AU_bargraphs      = True, If the script should create bar graphs and latter create a video with them . Bargraphs are visual representations of the blenshapes.
      export_blendshapes          = True; If the scripts should create the csv file with the analysis of the face (some people want to just generate videos and not generate the csv file—not many though).
      export_lmks              = False If the script should create a csv file with landmarks
      create_tracked_video     = True; If the script should create a tracked video from the tracked frames. This requires export_tracked_frames=True
      combine_AU_graphs_into_video = True ; If the script should combine all bargraphs into one video. This requires export_AU_bargraphs=True
      combine_AU_bargraphs_and_tracked_video = True; If the script should combine the bargraphs and tracked video together to create a video with both of the representations. This requires combine_AU_graphs_into_video = True, create_tracked_video  = True, export_tracked_frames=True, export_AU_bargraphs=True 
      fps            = 25 : Number of frames per second in the video. Usually 25.
      delete_frames  = True; If the script should deleted the frames after finishing so you can have only the videos (of the options to generate the videos are on). This is usefull because the frames take a lot of space.
      delete_bar_graphs = True; If the script should delete the bargraphs after finishing so you can have only the videos (of the options to generate the videos are on). This is useful because the frames take a lot of space.


    Concerning the "processing" files
    If you have very long videos, tou should use analyse_video instead of analyse_video_parallel. It will be easier to handle memory wise.
    But what you can do to perform computations in parallel, is that you can call that function several times on the HPC, the function will know which files the other functions currently running are being processed by checking inside the “processing” folder.

    If you want a file to be re-analysed either delete it from the analysis folder or from the processing folder.
  """
  
  print("Starting analysis for : " + source)

  file_tag = os.path.splitext(os.path.basename(source))[0]
  target_frames_folder   = target_frames_folder   + file_tag + "/"
  target_AU_plots_folder = target_AU_plots_folder + file_tag + "/"
  au_analysis_file = target_analysis_folder + file_tag + ".csv"
  lmk_analysis_file = target_analysis_folder + file_tag + "_lmks.csv"
  processing_file = target_processing_folder + file_tag + ".csv"

  #Create folders needed
  os.makedirs(target_processing_folder, exist_ok=True)

  if export_blendshapes or export_lmks:
    os.makedirs(target_analysis_folder, exist_ok=True)
  if export_tracked_frames:
    os.makedirs(target_frames_folder, exist_ok=True)

  if create_tracked_video:
    os.makedirs(target_video_folder, exist_ok=True)

  if export_AU_bargraphs:
    os.makedirs(target_AU_plots_folder, exist_ok=True)

  if combine_AU_graphs_into_video:
    os.makedirs(target_au_video_folder, exist_ok=True)

  if combine_AU_bargraphs_and_tracked_video:
    os.makedirs(combined_videos_folder, exist_ok=True)
  
  #Skip if analysis exists oor create processing file
  if os.path.isfile(au_analysis_file):
    print("Analysis already performed, skipping for : " + source)
    return
  elif os.path.isfile(processing_file):
    print("Analysis beeing done by someone else, skipping for : " + source)
    return
  else:
    #create processing file
    open(processing_file, "a")
    

  #General
  file_tag = ntpath.basename(source)

  #Mainscript
  BaseOptions = mp.tasks.BaseOptions
  FaceLandmarker = mp.tasks.vision.FaceLandmarker
  FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
  VisionRunningMode = mp.tasks.vision.RunningMode

  # Create a face landmarker instance with the video mode:
  options = FaceLandmarkerOptions(
          base_options=BaseOptions(model_asset_path=model_asset_path)
          , running_mode=VisionRunningMode.VIDEO
          , output_face_blendshapes = output_face_blendshapes
          , num_faces = num_faces
      )

  #Run detection on each frame of video
  detector = vision.FaceLandmarker.create_from_options(options)
  cap = cv2.VideoCapture(source)
  fps = cap.get(cv2.CAP_PROP_FPS)
  cpt=0
  tts = []
  detection_results = {}
  while cap.isOpened():
    if cap.grab():
      flag, frame = cap.retrieve()      
      
      #Compute time delay with last frame
      tt = int(cap.get(cv2.CAP_PROP_POS_MSEC))
      if int(tt) in tts:
        continue
      tts.append(tt)
      
      image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
      detection_result = detector.detect_for_video(image, timestamp_ms = tt)
      detection_results[cpt] = detection_result
      
      if export_tracked_frames and detection_result.face_landmarks:
        annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
        cv2.imwrite(target_frames_folder  +str(cpt)+".png", annotated_image)

      if export_AU_bargraphs and detection_result.face_blendshapes:
        fig = plot_face_blendshapes_bar_graph(detection_result.face_blendshapes[0])
        fig.savefig(target_AU_plots_folder+ str(cpt) + ".png")
        plt.close(fig)
      
      cpt+=1
    else:
      break      
  cap.release()
  print("Finished processing frames, processed "+ str(cpt) +" frames, for file : "+ file_tag)

  # Sanity checks
  #Check if something was done during analysis otherwise return
  if cpt==0:
    print("Exiting analysis because no frames were processed for file : "+ file_tag)
    return
  elif len(detection_results)==0:
    print("Exiting analysis because no detection results for file : "+ file_tag)
    return

  #Export analysis into a csv
  if export_blendshapes:
    print("Exporting blendhsapes")
    
    #Export blndshapes
    blend_df = pl.DataFrame()
    for frame in detection_results:
      detection_result = detection_results[frame]
      face_blendshapes = detection_result.face_blendshapes
      face_landmarks   = detection_result.face_landmarks

      #Create blendshape
      if face_blendshapes:
        for idx, category in enumerate(face_blendshapes[0]):
            category_name = category.category_name
            score = round(category.score, 2)
            df_aux = pl.DataFrame()
            df_aux = df_aux.with_columns(pl.lit(category_name).alias('blendshape'))
            df_aux = df_aux.with_columns(pl.lit(score).alias('score'))
            df_aux = df_aux.with_columns(pl.lit(frame).alias('frame'))
            blend_df = pl.concat([blend_df, df_aux])
    
    blend_df.write_csv(au_analysis_file)

  #Export lmks
  if export_lmks:
    print("Exporting lmks")
    lmks_df = pl.DataFrame()
    for frame in detection_results:
      detection_result = detection_results[frame]
      face_landmarks   = detection_result.face_landmarks
      # Export landmarks
      if face_landmarks:
        for idx, category in enumerate(face_landmarks[0]):
          x = category.x
          y = category.y
          z = category.z
          visibility = category.visibility
          presence = category.presence
          # Create a Polars DataFrame with the landmark name and frame number
          df_aux = pl.DataFrame({"idx": idx, "x": [x],"y": [y],"z": [z],"visibility":visibility ,"presence": presence , "frame": [frame]})
          # Concatenate the new rows to lmks_df
          lmks_df = pl.concat([lmks_df, df_aux])

    # Write the DataFrame to CSV
    lmks_df.write_csv(lmk_analysis_file)


  #Create video with frames
  if create_tracked_video:
    os.makedirs(target_video_folder, exist_ok=True)
    print("Creating Video")
    import uuid
    audio_file = str(uuid.uuid1()) +"____.wav"
    extract_audio(source, audio_file)
    create_movie_from_frames(frame_name_tag=target_frames_folder, fps=fps, img_extension=".png" , target_video = target_video_folder + file_tag, preset="ultrafast", audio_file=audio_file)

    if os.path.isfile(audio_file):
      os.remove(audio_file)

  if combine_AU_graphs_into_video:
    os.makedirs(target_video_folder, exist_ok=True)
    print("Creating Video")
    import uuid
    audio_file = str(uuid.uuid1()) +"____.wav"
    extract_audio(source, audio_file)
    create_movie_from_frames(frame_name_tag=target_AU_plots_folder, fps=fps, img_extension=".png" , target_video = target_au_video_folder + file_tag, preset="ultrafast")
    if os.path.isfile(audio_file):
      os.remove(audio_file)

  if combine_AU_bargraphs_and_tracked_video :
    left = target_au_video_folder + file_tag  
    right = target_video_folder + file_tag
    combine_2_videos(left, right, combined_videos_folder + file_tag, combine_audio_flag=False)

  #Delete frames if needed
  if delete_frames:
     print("Deleting frames")
     shutil.rmtree(target_frames_folder)
  
  #Delete frames if needed
  if delete_bar_graphs:
     print("Deleting bar graphs")
     shutil.rmtree(target_AU_plots_folder)

  #Delete processing file
  if os.path.isfile(processing_file):
    os.remove(processing_file)

  #Print finished
  print("Finished all for : " + source)


## Analyse folder parallel
def analyse_video_parallel(sources
                  , target_analysis_folder
                  , target_frames_folder     = "tracked/"
                  , target_video_folder      = "output_video/"
                  , target_au_video_folder   = "au_video/"
                  , target_AU_plots_folder   = "AU_bar_graph_folder/"
                  , combined_videos_folder   = "combined_videos_folder/"
                  , target_processing_folder = "processing/"
                  , model_asset_path         = "face_landmarker_v2_with_blendshapes.task"
                  , output_face_blendshapes  = True
                  , num_faces                = 1
                  , export_tracked_frames    = True
                  , export_AU_bargraphs      = True
                  , export_blendshapes          = True
                  , export_lmks              = False
                  , create_tracked_video     = True
                  , combine_AU_graphs_into_video = True
                  , combine_AU_bargraphs_and_tracked_video = True
                  , fps            = 25
                  , delete_frames  = True
                  , delete_bar_graphs = True
                    ):
  """
    source folder should be in the shape of glob.glob
    Usage example:
    from face_analysis_mp import analyse_video_parallel
    def main():
        analyse_video_parallel("processed/gst-1.22.6-chK/re-encode/*/*.mp4")

    if __name__ == '__main__':
        main()
    parameters:
      target_analysis_folder : The name of the csv folder where the csv file with the face analysis will be saved
      target_frames_folder  : The name of the folder where the frames will be saved (to create a video, we need to create different frames, with e.g. face tracking results)
      target_video_folder    : The name of the folder where the video with the tracking will be created.
      target_au_video_folder   : The name of the folder where the video that recreates faces from AUs will be created
      target_AU_plots_folder   : The name of the folder where the plots to create the video with AUs will be created
      combined_videos_folder  : The name of the folder where the combined videos (with tracking and AUs will be created)
      target_processing_folder :  The name of the folder where we will keep track f the videos being analysed at each time.
      model_asset_path            : Default : “face_landmarker_v2_with_blendshapes.task”. Name of the the mediapipe model to use
      output_face_blendshapes  : Default : True; If the model should compute blendhapes (e.g. smiling , frowning , etc...)
      num_faces                = 1; Number of faces to detect in the video
      export_tracked_frames    = True, If the script should export tracked frames.Tracked frames are visual representations of the faces with the landmarks within it..
      export_AU_bargraphs      = True, If the script should create bar graphs and latter create a video with them . Bargraphs are visual representations of the blenshapes.
      export_blendshapes          = True; If the scripts should create the csv file with the analysis of the face (some people want to just generate videos and not generate the csv file—not many though).
      export_lmks              = False. If the script should create a csv file with landmarks
      create_tracked_video     = True; If the script should create a tracked video from the tracked frames. This requires export_tracked_frames=True
      combine_AU_graphs_into_video = True ; If the script should combine all bargraphs into one video. This requires export_AU_bargraphs=True
      combine_AU_bargraphs_and_tracked_video = True; If the script should combine the bargraphs and tracked video together to create a video with both of the representations. This requires combine_AU_graphs_into_video = True, create_tracked_video  = True, export_tracked_frames=True, export_AU_bargraphs=True 
      fps            = 25 : Number of frames per second in the video. Usually 25.
      delete_frames  = True; If the script should deleted the frames after finishing so you can have only the videos (of the options to generate the videos are on). This is usefull because the frames take a lot of space.
      delete_bar_graphs = True; If the script should delete the bargraphs after finishing so you can have only the videos (of the options to generate the videos are on). This is useful because the frames take a lot of space.


    Concerning the "processing" files
    If you have very long videos, tou should use analyse_video instead of analyse_video_parallel. It will be easier to handle memory wise.
    But what you can do to perform computations in parallel, is that you can call that function several times on the HPC, the function will know which files the other functions currently running are being processed by checking inside the “processing” folder.

    If you want a file to be re-analysed either delete it from the analysis folder or from the processing folder.
  """

  os.makedirs(target_analysis_folder, exist_ok=True)
  os.makedirs(target_frames_folder, exist_ok=True)
  os.makedirs(target_video_folder, exist_ok=True)
  os.makedirs(target_au_video_folder, exist_ok=True)
  os.makedirs(target_AU_plots_folder, exist_ok=True)
  os.makedirs(combined_videos_folder, exist_ok=True)
  os.makedirs(target_processing_folder, exist_ok=True)


  import multiprocessing
  from itertools import repeat
  pool_obj = multiprocessing.Pool()
  sources     = glob.glob(sources)

  pool_obj.starmap(analyse_video, zip(sources
                                        , repeat(target_analysis_folder)
                                        , repeat(target_frames_folder)
                                        , repeat(target_video_folder)
                                        , repeat(target_au_video_folder)
                                        , repeat(target_AU_plots_folder)
                                        , repeat(combined_videos_folder)
                                        , repeat(target_processing_folder)
                                        , repeat(model_asset_path)
                                        , repeat(output_face_blendshapes)
                                        , repeat(num_faces)
                                        , repeat(export_tracked_frames)
                                        , repeat(export_AU_bargraphs)
                                        , repeat(export_blendshapes)
                                        , repeat(export_lmks)
                                        , repeat(create_tracked_video)
                                        , repeat(combine_AU_graphs_into_video)
                                        , repeat(combine_AU_bargraphs_and_tracked_video)
                                        , repeat(fps)
                                        , repeat(delete_frames)
                                        , repeat(delete_bar_graphs)
                                        ))