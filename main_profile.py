# !pip install -q git+https://github.com/google-deepmind/gemma.git

import os

os.environ["KAGGLE_USERNAME"] = 'cyrusxu5'
os.environ["KAGGLE_KEY"] = '0ddf9657f2ec99be6b26e51c22754f17'

import os
import enum
import re
import string

import chex
import jax
import jax.numpy as jnp
from jax.profiler import trace
import optax
import pickle

import tensorflow as tf

from gemma import params as params_lib
from gemma import sampler as sampler_lib
from gemma import transformer as transformer_lib

GEMMA_VARIANT = '2b-it' # @param ['2b', '2b-it'] {type:"string"}

import kagglehub

GEMMA_PATH = kagglehub.model_download(f'google/gemma/flax/{GEMMA_VARIANT}')
print('GEMMA_PATH:', GEMMA_PATH)
CKPT_PATH = os.path.join(GEMMA_PATH, GEMMA_VARIANT)
TOKENIZER_PATH = os.path.join(GEMMA_PATH, 'tokenizer.model')
print('CKPT_PATH:', CKPT_PATH)
print('TOKENIZER_PATH:', TOKENIZER_PATH)

# load gemma tokenizer
vocab = spm.SentencePieceProcessor()
vocab.Load(TOKENIZER_PATH)


#### #### #### ####   profile pre-trained  model #### #### #### #### 

params = params_lib.load_and_format_params(CKPT_PATH)
config_2b = transformer_lib.TransformerConfig.from_params(
    params,
    cache_size=30
)
model_2b = transformer_lib.Transformer(config=config_2b)

sampler = sampler_lib.Sampler(
    transformer=model_2b,
    vocab=vocab,
    params=params['params'],
)

# Profile the line using the pre-trained model
with trace() as pretrain_trace:
    pretrain_output = sampler(
        ["Translate this Chinese to Hindi:\n很高兴认识你\n"],
        total_generation_steps=100,
    ).text
print("Output (Pretrained Model):", pretrain_output)
print("Memory Usage (Pretrained Model):", pretrain_trace.trace_memory_usage())
print("CPU Time (Pretrained Model):", pretrain_trace.trace_main_thread_cpu_time())



#### #### #### ####  profile trained  model #### #### #### #### 

with open('params.pkl', 'rb') as f:
    loaded_params = pickle.load(f)

# Modify the code to use the loaded params
sampler = sampler_lib.Sampler(
    transformer=model_2b,
    vocab=vocab,
    params=loaded_params['params'],
)


# Profile the line using the trained model
with trace() as trained_trace:
    trained_output = sampler(
        ["Translate this Chinese to Hindi:\n很高兴认识你\n"],
        total_generation_steps=100,
    ).text
print("Output (Trained Model):", trained_output)
print("Memory Usage (Trained Model):", trained_trace.trace_memory_usage())
print("CPU Time (Trained Model):", trained_trace.trace_main_thread_cpu_time())



#### #### #### #### run main loop #### #### #### #### #### 

import speech.cloud_api as api
# Define the project ID and language codes
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
        translated_text = sampler(
            prompt,
            total_generation_steps=100,
        ).text
        print("Translated text:", translated_text)
        
        # Convert translated text to speech
        audio_output = api.cloud_TTS(translated_text, language_codes[lang])
        
        # Play the generated audio
        api.play_audio(audio_output)
    else:
        print("Language not supported for translation.")