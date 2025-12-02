#!/usr/bin/env python

# Plot scaling and wavelet functions for db, sym, coif, bior and rbio families

import itertools

import matplotlib.pyplot as plt

import pywt

plot_data = [('db', (2, 2)),
             ('sym', (2, 2)),
             ('coif', (2, 2))]


    # fig = plt.figure()
    # fig.subplots_adjust(hspace=0.2, wspace=0.2, bottom=.02, left=.06,
    #                     right=.97, top=.94)
for (wave, color) in [("haar", "b"), ("db2", "g"), ("sym3", "r"), ("coif1", "c")]:

    wavelet = pywt.Wavelet(wave)
    phi, psi, x = wavelet.wavefun(level=5)

            # ax = fig.add_subplot(rows, 2 * cols, 1 + 2 * (col + row * cols))
            # ax.set_title(wavelet.name + " phi")
            # ax.plot(x, phi, color)
            # ax.set_xlim(min(x), max(x))

    fig, ax = plt.subplots(1, 1)
    # ax = fig.add_subplot(rows, 2*cols, 1 + 2*(col + row*cols) + 1)
    n = wavelet.name if wavelet.name != "db1" else "haar"
    ax.set_title(n)
    ax.plot(x, psi, color)
    ax.set_xlim(min(x), max(x))
    plt.savefig("wavelets/" + n + ".pdf")


# plt.show()