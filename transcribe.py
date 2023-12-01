#execture with : conda activate whisper

import os
import glob
from conversions import get_file_without_path
from video_processing import extract_audio
import whisper
from transform_audio import extract_sentences_tags


## Parallel processing functions
def transcribe_video_file(file, transcription_path="transcribed/", audio_path= "extracted_audio/", model_type= "large", language="English", fp16=True, extract_audio=True):
    print("Analysing : " + file)
    
    file_tag = get_file_without_path(file)
    target_path = transcription_path + file_tag + ".txt"
    if os.path.isfile(target_path):
        print("Skipping this file because it exists already  : " + target_path)
        return
    
    #Create empty results file, to say that we are in the process of anlysing it
    open(target_path, "a")

    #handle file names
    file_tag = get_file_without_path(file)
    audio = audio_path + file_tag+".wav"
    
    #Extract audio
    if extract_audio:
        extract_audio(file, audio)
    
    #Speech to text
    model = whisper.load_model(model_type)
    result = model.transcribe(audio, fp16=fp16, language=language)
    output = result["text"]
    print(output)
    
    #Write results
    text_file = open(target_path, "w")
    text_file.write(output)
    text_file.close()


def transcribe_video_file_time_stamps(file, transcription_path="transcribed/", audio_path= "extracted_audio/", model_type= "large", language="English", fp16=True, device="cpu", extract_audio=True):
    import whisper_timestamped as whisper
    import json

    print("Analysing : " + file)
    
    file_tag = get_file_without_path(file)
    target_path = transcription_path + file_tag + ".txt"
    if os.path.isfile(target_path):
        print("Skipping this file because it exists already  : " + target_path)
        return
    
    #Create empty results file, to say that we are in the process of anlysing it
    open(target_path, "a")

    #handle file names
    file_tag = get_file_without_path(file)
    audio_file = audio_path + file_tag+".wav"

    #Extract audio
    if extract_audio:
        extract_audio(file, audio_file)
    
    #Speech to text
    audio = whisper.load_audio(audio_file)
    model = whisper.load_model(model_type, device=device)
    result = whisper.transcribe(model, audio, language="fr")
    output = json.dumps(result, indent = 2, ensure_ascii = False)
    print(output)
    
    #Write results
    text_file = open(target_path, "w")
    text_file.write(output)
    text_file.close()


def transcribe_wav_file(file, transcription_path="transcribed/", audio_path= "extracted_audio/", model_type= "large", language="English", fp16=True):
    print("Analysing : " + file)
    

    file_tag = get_file_without_path(file)
    target_path = transcription_path + file_tag + ".txt"
    if os.path.isfile(target_path):
        print("Skipping this file because it exists already  : " + target_path)
        return
    
    #Create empty results file, to say that we are in the process of anlysing it
    open(target_path, "a")
    
    #Speech to text
    model = whisper.load_model(model_type)
    result = model.transcribe(audio, fp16=fp16, language=language)
    output = result["text"]
    print(output)
    
    #Write results
    text_file = open(target_path, "w")
    text_file.write(output)
    text_file.close()


def transcribe_parallel_time_stamps(sources, transcription_path="transcribed/", audio_path="extracted_audio/", model_type="large", language="English", fp16=False, device="cpu", extract_audio=True):
    """
    source folder should be in the shape of glob.glob
    audio_path : is where the audio will be stored after extraction
    Here for more details : https://github.com/openai/whisper

    Usage example:
        Add description here
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
    
    pool_obj.starmap(transcribe_video_file_time_stamps, zip(sources
                                               , repeat(transcription_path)
                                               , repeat(audio_path)
                                               , repeat(model_type)
                                               , repeat(language)
                                               , repeat(fp16)
                                               , repeat(device)
                                               , repeat(extract_audio)
                                               ))

def transcribe_parallel(sources, transcription_path="transcribed/", audio_path="extracted_audio/", model_type="large", language="English", fp16=False, extract_audio=True):
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
                                               , repeat(extract_audio)
                                               ))

