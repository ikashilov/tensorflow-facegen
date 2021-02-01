#!/usr/bin/python
# -*- coding: latin-1 -*-
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

plt.rcParams['xtick.bottom'] = False
plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['ytick.left'] = False
plt.rcParams['ytick.labelleft'] = False


NOISE_DIM = 100
IMG_SHAPE = (36, 36, 3)


def sample_noise_batch(bsize):
    return np.random.normal(size=(bsize, NOISE_DIM)).astype('float32')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nrow', type=int, default=8, help='number of rows to generate. final image is nrow*ncol')
    parser.add_argument('--ncol', type=int, default=8, help='number of columns to generate.  final image is nrows*ncol')
    parser.add_argument('--path', type=str, default='saved_model/my_model', help='path to the saved genrator model')
    parser.add_argument('--save', type=bool, default=False, help='save image')
    args = parser.parse_args()

    if args.nrow < 1 or args.ncol < 1:
        exit(f'cannot generate image {args.nrow} x {args.ncol}')
    try:
        model = tf.keras.models.load_model(args.path, compile=False)
    except Exception as e:
        exit(f'cannot load the model: {e}')

    images = model.predict(sample_noise_batch(bsize=args.nrow*args.ncol)).clip(0, 1)

    plt.figure(figsize=(args.nrow, args.ncol))

    for i in range(args.nrow * args.ncol):
        plt.subplot(args.nrow, args.ncol, i + 1)
        plt.imshow(images[i].reshape(IMG_SHAPE), cmap='gray')

    if args.save:
        plt.savefig(f'./image_{args.nrow}x{args.ncol}.png')

    plt.show()


if __name__ == "__main__":
    main()
