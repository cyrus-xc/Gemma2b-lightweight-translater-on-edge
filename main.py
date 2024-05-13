from pygemma import Gemma
import speech.cloud_api as api 


gemma = Gemma()
# gemma.show_help()
# gemma.show_config()
gemma.load_model("/path/to/tokenizer", "/path/to/compressed_weight/", "model_type")

project_id = "gemma-speech-420819"
language_codes = ["zh-Hans-CN", "hi-IN"]

# Main loop
while True:
    duration = 5  # seconds
    audio_file = "outputs/audio.wav"  # Define the path to save recorded audio
    api.record_audio(duration, audio_file)  # Record audio
    print("Transcribing audio...")
    response, languages = api.cloud_STT(audio_file)  # Transcribe speech to text
    print("Transcribed text:", response)
    print("Detected languages:", languages)
    
    # Determine the translation prompt based on the detected language
    lang = 0
    if "hi-IN" in languages:
        lang = 0
        prompt = f"Translate this Hindi to Chinese:\n{response}\n"
    elif "zh-Hans-CN" in languages:
        lang = 1
        prompt = f"Translate this Chinese to Hindi:\n{response}\n"
    else:
        prompt = None
    
    if prompt:
        # Feed the prompt to the model and get the translated text
        translated_text = gemma.completion(prompt)
        print("Translated text:", translated_text)
        
        # Convert translated text to speech
        audio_output = api.cloud_TTS(translated_text, language_codes[lang])
        
        # Play the generated audio
        api.play_audio(audio_output)
    else:
        print("Language not supported for translation.")