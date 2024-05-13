from pygemma import Gemma, ModelType, ModelTraining
import speech.cloud_api as api
import re
import time

# gemma.show_help()
# gemma.show_config()
gemma = Gemma(
    tokenizer_path="gemma-cpp/build/tokenizer.spm",
    compressed_weights_path="gemma-cpp/build/2b-it-sfp.sbs",
    # compressed_weights_path="gemma-cpp/build/params.pkl",
    model_type=ModelType.Gemma2B,
    model_training=ModelTraining.GEMMA_IT,
)

project_id = "gemma-speech-420819"
language_codes = ["zh-Hans-CN", "hi-IN"]

# Main loop
while True:
    api.beep()
    duration = 4
    audio_file = "outputs/audio.wav"  # Define the path to save recorded audio
    api.record_audio(duration, audio_file)  # Record audio
    start_time = time.time()
    print("Transcribing audio...")
    response, languages = api.cloud_STT(audio_file)  # Transcribe speech to text
    print("Transcribed text:", response)
    print("Detected languages:", languages)

    # Determine the translation prompt based on the detected language
    lang = 0

    if "hi-in" in languages:
        lang = 0
        # prompt = f"Translate this Hindi to Chinese, don't give any explaination:\n{response[0]}\n"
        prompt = f"Translate this Hindi sentence to English and Chinese in this format ('chinese': ' ', 'english: ' ')\n{response[0]}\n"
    elif "zh-Hans-CN" in languages or "cmn-hans-cn" in languages:
        lang = 1
        # prompt = f"Translate this Chinese to Hindi, don't give any explaination:\n{response[0]}\n"
        prompt = f"Translate this Chinese sentence to English and Hindi in this format ('hindi': ' ', 'english: ' ')\n{response[0]}\n"
    else:
        prompt = None

    if prompt:
        # Feed the prompt to the model and get the translated text
        start_gemma_time = time.time()
        translated_text = gemma(prompt) + "\n"
        end_gemma_time = time.time()
        print("Translated text:", translated_text, "\n in ", end_gemma_time-start_gemma_time, "seconds\n")

        # Convert translated text to speech
        matches = ""
        if lang == 0:
            pattern_chinese = r"\*\*Chinese:\*\* (.+)\n"
            matches = re.findall(pattern_chinese, translated_text, re.DOTALL)
        else:
            pattern_hindi = r"\*\*Hindi:\*\* (.+)\n"
            matches = re.findall(pattern_hindi, translated_text, re.DOTALL)

        audio_output = api.cloud_TTS("output" + matches[0], language_codes[lang])

        # Play the generated audio
        end_time = time.time()
        api.play_audio(audio_output)
        
        # measure time taken for whole loop
        print("Time taken for a whole loop:", end_time - start_time, "seconds\n\n")
    else:
        print("Language not supported for translation.")
