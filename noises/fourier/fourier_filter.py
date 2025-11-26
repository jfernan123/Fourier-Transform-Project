from scipy import fftpack
import numpy as np

def fourier_denoise(image, percentile=85, K=50):
    # 1. FFT
    F = fftpack.fft2(image)
    F_shift = fftpack.fftshift(F)  # center DC
    H, W = image.shape

    # 2. Magnitude of centered spectrum
    mag = np.abs(F_shift)

    # 3. Optionally zero a small DC block (low frequencies)
    mag[H//2 - K:H//2 + K, W//2 - K:W//2 + K] = 0

    # 4. Keep coefficients below some percentile (remove large peaks)
    thresh = np.percentile(mag, percentile)
    mask = mag < thresh  # Keep the small coefficients as we only care about the noise right now, however we can't remove too many small coefficients.
    # 5. Apply mask in centered domain
    F_shift_filtered = F_shift * mask

    # 6. Unshift and inverse FFT
    F_filtered = fftpack.ifftshift(F_shift_filtered)
    image_filtered = np.real(fftpack.ifft2(F_filtered))

    return image_filtered
