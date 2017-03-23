# Blekota
Graduation project at [Gymnázium Jana Keplera](http://gjk.cz).

Blekota uses recurrent neural networks to predict how a given training sound continues based on the sound's previous data. We can generate new sounds by letting Blekota predict the continuation of a sound, append the prediction to the sound and repeat the process.

Blekota is inspired by Andrej Karpathy's [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/), where his model generates text similar to a training file. Blekota takes this concept and applies it to sound. Also, here we use GRUs instead of LSTMs.

## Installation
This is roughly how to install Blekota on Debian-based Linux. Your mileage may vary.

1. Install Python 3 (with development packages: `python3.4-dev`) and pip

2. Install non-Python dependencies:
    `sudo apt-get install python3-cffi libportaudio2 libsndfile-dev`

3. Install Python dependencies:
    `sudo pip install numpy pysoundfile pysoundcard matplotlib`

## Sampling from an existing model

Now Blekota should be ready. To try it out, download [the sample model](https://drive.google.com/open?id=0B3cbiAacE6JcZFNxNFg4RVVRXzQ) into `samples/bety.pkl` and run the following from the project root:

```bash
python3 -i src/blekota.py samples/bety.pkl
```

This loads the model and enters the Python REPL.
Now you can do various things with the model. Let's try:

```python
>>> y2 = clf.sample(8000)    # generate 8000 samples (one second) of sound
>>> play(y2)                 # play the generated sound
>>> save_file('foo.wav', y2) # save the sound into foo.wav
```

If all goes well, the sound should be one second of singing. There is a slight chance that the generated sound will just be silence, if this happens, try again or generate a longer sample.

## Creating and training a new model

To create your own model, run:

```bash
python3 -i src/blekota.py --model-name my_model my_training_file.wav
```
`my_model` is the name of the model - it will be saved into files prefixed with `my_model`. For example if we set `--model-name foo/bar` then the model will be saved into files in the form of `foo/bar_N.pkl`, where `N` is the number of iterations the model was trained for.

`my_training_file.wav` is the file the model will train on; it will attempt to create sounds similar to those in the training file. The file should be a `.wav`. By default, Blekota assumes the sampling frequency is 8 kHz. Certain functions (anything where the timescale is relevant: `play`, `show`, `save_file`) require setting the sampling frequency manually through the named argument `fs`. You can also change the default in `const.py`.

`blekota.py` sets reasonable defaults for hyperparameters. For a detailed description, see the Hyperparameters section.

After the model has been created, we enter the Python REPL. You can run the following ocmmands:

* `clf.train(it)` - train the model for `it` iterations. Stopping the training through `ctrl+C` does not break the model, so it is possible to set `it` to a high number and stop the training at any time (when it seems to stagnate). The model saves automatically every 1000 iterations.
* `clf.sample(n)` - generate `n` samples of sound. Returns a NumPy array of length `n` array with the generated sound.
* `clf.checkpoint(n)` - save the model, generate and save `n` samples of sound. No samples are taken when `n==0`.
* `play(sound)` - play a sound saved in the NumPy array `sound`. The played sound is saved into `last_played.wav`; this is the simplest way to save the generated sounds.
* `show(sound)` - plot the sound saved in `sound`.
* `heatmap(start, length)` - display a heatmap of the last generated sound. Visualises `length` samples beginning from `start`.
* `save_file(file, sound)` - save `sound` into the file at `file`. `file` should end in `.wav`.

A few more functions are available and some of the listed functions have more advanced usage (especially `sample`, which allows changing the temperature and giving a "hint"). These functions are documented in the code itself.

## Hyperparameters
To see a brief description of which hyperparameters can be changed from `blekota.py`, run `python3 src/blekota.py --help`. Here is an intuition about what each hyperparameter does:

`--layers` - the number of layers of the model. Using multiple layers is a good way to make the model more powerful without using too many resources - memory and running time are both linear in the number of layers. Default: 3

`--hidden` - the size of the hidden vector of each layer. Should probably be on the order of several hundred - too small makes the model weak, too large makes the computation slow and expensive in memory. Default: 256

`--seq-length` - the number of steps of the model to perform before backpropagating and updating parameters. Increasing makes the model train slower and use more memory (many computed values are cached to make backpropagation faster), but should make the model realize more long-term dependencies, which, especially in sound, is crucial. Default: 100

`--batch-size` - the size of mini-batches used in gradient descent.  `batch-size` sequences are computed simultaneously. A higher batch size is more time-efficient because we multiply larger matrices, but we run into memory constraints. Also, a larger batch size means less fluctuation in cost throughout training. Default: 80