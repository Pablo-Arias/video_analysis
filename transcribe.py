#execture with : conda activate whisper

import os
import glob
from conversions import get_file_without_path
from video_processing import extract_audio
import whisper
from transform_audio import extract_sentences_tags


## Parallel processing functions
def transcribe_video_file(file, transcription_path="transcribed/", audio_path= "extracted_audio/", model_type= "large", language="English", fp16=True, extract_audio_flag=True):
    print("Analysing : " + file, flush=True)
    
    file_tag = get_file_without_path(file)
    target_path = transcription_path + file_tag + ".txt"
    if os.path.isfile(target_path):
        print("Skipping this file because it exists already  : " + target_path, flush=True)
        return
    
    #Create empty results file, to say that we are in the process of anlysing it
    open(target_path, "a")

    #handle file names
    file_tag = get_file_without_path(file)
    audio = audio_path + file_tag+".wav"
    
    #Extract audio
    if extract_audio_flag:
        extract_audio(file, audio)
    
    #Speech to text
    model = whisper.load_model(model_type)
    result = model.transcribe(audio, fp16=fp16, language=language)
    output = result["text"]
    print(output, flush=True)
    
    #Write results
    text_file = open(target_path, "w")
    text_file.write(output)
    text_file.close()


def transcribe_video_file_time_stamps(file, transcription_path="transcribed/"
                                      , audio_path= "extracted_audio/"
                                      , model_type= "large"
                                      , language="En"
                                      , device="cpu"
                                      , extract_audio_flag=True
                                      , temperature = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
                                      , best_of=5
                                      , beam_size=5
                                      , vad="auditok"
                                      , detect_disfluencies=True
                                      ):
    """
    Check transcription arguiments here: https://github.com/linto-ai/whisper-timestamped#light-installation-for-cpu
    """
    import whisper_timestamped as whisper
    import json

    print("Analysing : " + file, flush=True)
    
    file_tag = get_file_without_path(file)
    target_path = transcription_path + file_tag + ".txt"
    if os.path.isfile(target_path):
        print("Skipping this file because it exists already  : " + target_path, flush=True)
        return
    
    #Create empty results file, to say that we are in the process of anlysing it
    open(target_path, "a")

    #handle file names
    file_tag = get_file_without_path(file)
    audio_file = audio_path + file_tag+".wav"

    #Extract audio
    if extract_audio_flag:
        extract_audio(file, audio_file)
    
    #Speech to text
    audio = whisper.load_audio(audio_file)
    model = whisper.load_model(model_type, device=device)

    
    result = whisper.transcribe(model, audio, language=language, beam_size=beam_size, best_of=best_of, temperature=temperature, vad=vad, detect_disfluencies=detect_disfluencies)

    output = json.dumps(result, indent = 2, ensure_ascii = False)
    print(output, flush=True)
    
    #Write results
    text_file = open(target_path, "w")
    text_file.write(output)
    text_file.close()


def transcribe_wav_file(file, transcription_path="transcribed/", audio_path= "extracted_audio/", model_type= "large", language="English", fp16=True):
    print("Analysing : " + file, flush=True)
    

    file_tag = get_file_without_path(file)
    target_path = transcription_path + file_tag + ".txt"
    if os.path.isfile(target_path):
        print("Skipping this file because it exists already  : " + target_path, flush=True)
        return
    
    #Create empty results file, to say that we are in the process of anlysing it
    open(target_path, "a")
    
    #Speech to text
    model = whisper.load_model(model_type)
    result = model.transcribe(audio, fp16=fp16, language=language)
    output = result["text"]
    print(output, flush=True)
    
    #Write results
    text_file = open(target_path, "w")
    text_file.write(output)
    text_file.close()


def transcribe_parallel_time_stamps(sources, transcription_path="transcribed/"
                                    , audio_path="extracted_audio/"
                                    , model_type="large"
                                    , language="English"
                                    , device="cpu"
                                    , extract_audio_flag=True
                                    , temperature = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
                                    , best_of = 5
                                    , beam_size = 5
                                    , vad = "auditok"
                                    , detect_disfluencies = True
                                    ):
    """
    source folder should be in the shape of glob.glob
    audio_path : is where the audio will be stored after extraction
    And here for parameteres : https://github.com/linto-ai/whisper-timestamped#light-installation-for-cpu

    Usage example:
        Add description here
    """
    os.makedirs(transcription_path, exist_ok=True)
    os.makedirs(audio_path, exist_ok=True)

    import multiprocessing
    from itertools import repeat
    
    pool_obj = multiprocessing.Pool()
    sources     = glob.glob(sources)
    
    pool_obj.starmap(transcribe_video_file_time_stamps, 
                                      zip(sources
                                      , repeat(transcription_path)
                                      , repeat(audio_path)
                                      , repeat(model_type)
                                      , repeat(language)
                                      , repeat(device)
                                      , repeat(extract_audio_flag)
                                      , repeat(temperature)
                                      , repeat(best_of)
                                      , repeat(beam_size)
                                      , repeat(vad)
                                      , repeat(detect_disfluencies)
                                        )
                    )
    
    print("Done transcribing parallel!", flush=True)


## Transcrive parallel
def transcribe_parallel(sources, transcription_path="transcribed/", audio_path="extracted_audio/", model_type="large", language="English", fp16=False, extract_audio_flag=True):
    """
    source folder should be in the shape of glob.glob
    audio_path : is where the audio will be stored after extraction
    Here for more details : https://github.com/openai/whisper

    Usage example:
        from transcribe import transcribe_parallel
        def main():
            transcribe_parallel("processed/gst-1.22.6-chK/re-encode/*/*.mp4", model_type="small")

        if __name__ == '__main__':
            transcribe_parallel()
    """
    try:
        os.mkdir(transcription_path)
        os.mkdir(audio_path)
    except:
        pass

    import multiprocessing
    from itertools import repeat
    
    pool_obj = multiprocessing.Pool()
    sources     = glob.glob(sources)
    
    pool_obj.starmap(transcribe_video_file, zip(sources
                                               , repeat(transcription_path)
                                               , repeat(audio_path)
                                               , repeat(model_type)
                                               , repeat(language)
                                               , repeat(fp16)
                                               , repeat(extract_audio_flag)
                                               ))

def generate_subtitles(target_folder, transcription_folder):
    import json
    import datetime

    #create target folders
    os.makedirs(target_folder, exist_ok=True)

    for transcription_file in glob.glob(transcription_folder):
        #Prepare variables
        file_tag = get_file_without_path(transcription_file)
        target_srt_file = target_folder + file_tag + ".srt"

        #Open transcription
        f = open(transcription_file)
        data = json.load(f)

        #Remove srt file if it exists already
        try:
            os.remove(target_srt_file)
        except OSError:
            pass
        
        #Create new file
        srt_file = open(target_srt_file, "a")

        # create new file with srt format
        for segment in data["segments"]:
            #Extract data
            id    = segment["id"]
            text  = segment["text"]
            start = segment["start"]
            start = str(datetime.timedelta(seconds=start))
            end   = segment["end"]
            end = str(datetime.timedelta(seconds=end))
            
            #write it
            srt_file.write(str(id) + "\n")
            srt_file.write(start + "-->" + end + "\n")
            srt_file.write(text + "\n")
            srt_file.write("\n")
        
        srt_file.close()