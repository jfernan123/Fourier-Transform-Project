
import numpy as np
from scipy import fftpack
from scipy.ndimage import uniform_filter, gaussian_filter

def fourier_wiener_denoise(image, noise_var=None, smooth_psd_size=6, eps=1e-8):
    img = np.asarray(image, dtype=np.float64)
    H, W = img.shape

    F = fftpack.fft2(img)
    F_shift = fftpack.fftshift(F)

    S_y = np.abs(F_shift) ** 2
    S_y_smooth = uniform_filter(S_y, size=smooth_psd_size)

    #Because noise is at the corners we estimate 
    #the variance of the noise by taking the median
    if noise_var is None:
        h_c = max(1, H // 8)
        w_c = max(1, W // 8)
        corners = np.concatenate([
            S_y[-h_c:, -w_c:].ravel(),
            S_y[-h_c:, :w_c].ravel(),
            S_y[:h_c, -w_c:].ravel(),
            S_y[:h_c, :w_c].ravel()
        ])
        noise_var = np.mean(corners) #Try to estimate the noise
    S_n = noise_var
    #Clean signal with noise remove
    S_x_hat = np.maximum(S_y_smooth - S_n, 0.0)
    H_wiener = S_x_hat / (S_x_hat + S_n + eps) #Obtain the wiener filter plus add an epsilon for some numerical stability

    F_filt = F_shift * H_wiener
    img_rec = np.real(fftpack.ifft2(fftpack.ifftshift(F_filt)))

    return img_rec.clip(0, 255).astype(np.uint8)

