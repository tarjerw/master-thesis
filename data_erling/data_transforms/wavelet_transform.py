import pywt

def wavelet_transform_data(data, mother_wavelet='coif1', n_levels=2):
    data  = data[:200].to_numpy()
    fig, axs = plt.subplots(3)
    axs[0].plot(data, c='g')
    #Compared to the dwt and idwt, these are naturally multi-level, and does not have to be conducted multiple times
    coeffs = pywt.wavedec(data, mother_wavelet, level=n_levels) #Mother wavelet type can be found via pywt.families() - dimension must be added at the end
    approx = coeffs[0]
    details = coeffs[1:]
    #sigma = np.mean( np.absolute( coeffs - np.mean(coeffs[-n_levels], axis=0) ), axis=0 )
    #uthresh = sigma * np.sqrt(2 * np.log(len(data)))
    #coeffs[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeffs[1:])
    result = pywt.waverec(coeffs, mother_wavelet) #Returns the same signal if all coefficients are used, since cA_n is the residual after the wavelet coefficients
    axs[0].plot(approx, c='r')
    axs[1].plot(data, c='g')
    axs[2].plot(approx, c='r')
    plt.show()

wavelet_transform_data(training_data)
