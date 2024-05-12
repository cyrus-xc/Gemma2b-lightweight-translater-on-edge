from typing import List
import sounddevice as sd
import scipy.io.wavfile as wav
import time
import os

# To use cloud api, please first login to the account
# 1. install gcloud cli: https://cloud.google.com/docs/authentication/provide-credentials-adc?hl=zh-cn#local-dev
# 2. create login token with command `gcloud auth application-default login`

# Auto detect speech language and transcript to text
from google.cloud import speech_v1p1beta1 as speech

def cloud_STT(speech_file):
    client = speech.SpeechClient()

    with open(speech_file, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,
        audio_channel_count=2,
        language_code="en-US",
        alternative_language_codes=["zh-Hans-CN", "hi-IN"],
    )

    start_time = time.time()
    response = client.recognize(config=config, audio=audio)
    end_time = time.time()
    print("Time taken:", end_time - start_time, "s")
    transcripts = []

    for i, result in enumerate(response.results):
        alternative = result.alternatives[0]
        print("-" * 20)
        print(f"First alternative of result {i}: {alternative}")
        print(f"Transcript: {alternative.transcript}")
        transcripts.append(alternative.transcript)

    return transcripts



from google.cloud import texttospeech

def cloud_TTS(text, language="en-US"):
    client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code=language, ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )

    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    with open("output.mp3", "wb") as out:
        out.write(response.audio_content)
    
    print("Text-to-Speech conversion completed. Output saved as output.mp3")

    return "output.mp3"




def record_audio(duration, file_path):
    print("Recording...")
    fs = 44100
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=2, dtype='int16')
    sd.wait()
    print("Recording finished.")
    wav.write(file_path, fs, recording)


# from google.cloud.speech_v2 import SpeechClient
# from google.cloud.speech_v2.types import cloud_speech
# def transcribe_multiple_languages_v2(
#     project_id: str,
#     language_codes: List[str],
#     audio_file: str,
# ) -> cloud_speech.RecognizeResponse:
#     """Transcribe an audio file."""
#     # Instantiates a client
#     client = SpeechClient()

#     # Reads a file as bytes
#     with open(audio_file, "rb") as f:
#         content = f.read()

#     config = cloud_speech.RecognitionConfig(
#         auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
#         language_codes=language_codes,
#         model="latest_short",
#     )

#     request = cloud_speech.RecognizeRequest(
#         recognizer=f"projects/{project_id}/locations/global/recognizers/_",
#         config=config,
#         content=content,
#     )

#     # Transcribes the audio into text
#     response = client.recognize(request=request)

#     for result in response.results:
#         print(f"Transcript: {result.alternatives[0].transcript}")

#     return response





############################
###### Example useage #########
############################

project_id = "gemma-speech-420819"
language_codes = ["zh-TW", "hi-IN"]

for i in range(1):
    duration = 5  # seconds
    audio_file = f"outputs/audio_{i}.wav"
    # record_audio(duration, audio_file)
    print(f"Transcribing audio {i+1}")
    response = cloud_STT(audio_file)
    
    # Convert transcribed text to speech
    for text in response:
        print(f"Converting text to speech for: {text}")
        audio_output = cloud_TTS(text)





