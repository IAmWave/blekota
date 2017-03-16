import argparse

parser = argparse.ArgumentParser(description='Load or create a Blekota model')

parser.add_argument('file',
                    help='Either a .pkl file containing an existing model or a .wav sound on which to train a new model. If a .wav is provided, other parameters can be used to specify model hyperparameters.')
# parser.add_argument('--model-file',
#                    +help='Load an existing model from this file. If unspecified, new model will be created')
# parser.add_argument('--sound-file',
#                    help='The sound file to train on. Should be a .wav')
parser.add_argument('--model-name', default='unnamed_model',
                    help='A name for the model, it will later be saved in files like so: [MODEL_NAME]_123.pkl')
parser.add_argument('--layers', type=int, default=3,
                    help='The number of layers of the model')
parser.add_argument('--hidden', type=int, default=256,
                    help='The size of the hidden vector of each layer')
parser.add_argument('--seq-length', type=int, default=100,
                    help='Number of steps to perform before updating parameters')
parser.add_argument('--batch-size', type=int, default=80,
                    help='Number of sequences to process in parallel')
parser.add_argument('--alpha', type=float, default=0.002,
                    help='Learning rate. Do not change if unsure')

args = parser.parse_args()

import numpy as np
import pickle
import audio_io
import rnn
import gru
import visual


clf = None

if args.file.endswith('.pkl'):  # load existing model
    print('Loading model from', args.file)
    with open(args.file, 'rb') as f:
        clf = pickle.load(f)
else:
    print('Creating new model')
    if not args.file.endswith('.wav'):
        exit('file should be either a .wav (create new model) or .pkl (load existing model)')

    clf = gru.GRU(args.hidden, layer_n=args.layers, seq_length=args.seq_length, batches=args.batch_size,
                  alpha=args.alpha, name=args.model_name, file=args.file)


def heatmap(start=0, length=1000):
    """After sampling a sound, visualise a part of it as a heatmap."""
    visual.heatmap(clf.p[start:(start + length)])

show = visual.show  # visualise a sound
play = audio_io.play  # play a sound

print(clf)
