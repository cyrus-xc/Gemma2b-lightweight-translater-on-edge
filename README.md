# Gemma2b-voice-translator-on-edge

This project is to build a real-time voice translator based on Google Gemma 2b LLMon edge computing devices like Raspberry Pi. 





## How to run

*Cyrus: example guide for running gemma-cpp ONLY. We may want to modify it to our project scope*

*Original Repo: https://github.com/google/gemma.cpp*

##### Step 1: Obtain model weights and tokenizer from Kaggle

Visit [the Gemma model page on Kaggle](https://www.kaggle.com/models/google/gemma/frameworks/gemmaCpp) and select `Model Variations |> Gemma C++`. On this tab, the `Variation` dropdown includes the options below. Download model `2b-it-sfp` 

##### Step 2: Extract and move file

Extract files from `archive.tar.gz` :

```
tar -xf archive.tar.gz
```

This should produce a file containing model weights such as `2b-it-sfp.sbs` and a tokenizer file (`tokenizer.spm`). Then move them to `gemma-cpp/build/` directory.

##### Build

We will use `cmake` to build the weights for `gemma-cpp`. Make sure to change your working directory to `gemma-cpp`, and then run the `cmake` invocations:

```
cd gemma-cpp
cmake -B build
```

After running whichever of the above `cmake` invocations that is appropriate for your weights, you can enter the `build/` directory and run `make` to build the `./gemma` executable:

```
cd build

# Configure `build` directory
cmake --preset make

# Build project using make
cmake --build --preset make -j8
```

`make -j8 gemma` will build using 8 threads, adjust the number according to your system.

If the build is successful, you should now have a `gemma` executable in the `build/` directory.

##### Step 4: Run

You can now run `gemma` from inside the `build/` directory.

```
./gemma \
--tokenizer tokenizer.spm \
--compressed_weights 2b-it-sfp.sbs \
--model 2b-it
```