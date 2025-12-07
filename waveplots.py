import itertools
import matplotlib.pyplot as plt
import pywt

plot_data = [('db', (2, 2)),
             ('sym', (2, 2)),
             ('coif', (2, 2))]

for (wave, color) in [("haar", "b"), ("db2", "g"), ("sym3", "r"), ("coif1", "c")]:

    wavelet = pywt.Wavelet(wave)
    phi, psi, x = wavelet.wavefun(level=5)

    fig, ax = plt.subplots(1, 1)
    n = wavelet.name if wavelet.name != "db1" else "haar"
    ax.set_title(n)
    ax.plot(x, psi, color)
    ax.set_xlim(min(x), max(x))
    plt.savefig("wavelets/" + n + ".pdf")
