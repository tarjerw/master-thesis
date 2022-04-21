import pywt
from data_processing import training_data, test_data
import matplotlib.pyplot as plt

#This now basically contains versiona I have tried but not had the heart to delete, of wavelet transforms.


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



'''

def wavelet_transform_data(data, mother_wavelet='coif1', n_levels=40):
    data  = data.to_numpy()
    fig, axs = plt.subplots(3)

    axs[2].plot(data)
    axs[2].set_title('original and cleaned')       

    axs[0].plot(data)
    axs[0].set_title('original')


    coefficients = pywt.wavedec(data, mother_wavelet, level=n_levels, mode='per')
    sigma = mad(coefficients[-1])
    thresh = sigma * np.sqrt(2 * np.log(len(data)))
    coefficients[1:] = (pywt.threshold(i, value=thresh, mode='hard') for i in coefficients[1:])
    clean = pywt.waverec(coefficients, mother_wavelet, mode='per')

    axs[1].plot(clean)
    axs[1].set_title('Cleaned')
    #axs[0].plot(clean)

    axs[2].plot(clean)

    plt.show()

'''

'''
def wavelet_transform_data(data, mother_wavelet='coif1'):
    
    fig, axs = plt.subplots(3)
    axs[0].plot(data, c='g')
    axs[0].set_title('original')

    cA, cD = pywt.dwt(data, mother_wavelet)

    result = pywt.idwt(cA, cD, mother_wavelet)

    print(len(cD))

    axs[0].plot(cA, c='b')
    axs[0].set_title('Original and cA')

    axs[1].plot(cA, c='b')
    axs[1].set_title('cA')

    axs[2].plot(cD)
    axs[2].set_title('cD')

    plt.show()
    
'''



'''

def wavelet_transform_data(data, mother_wavelet='coif1', n_levels=2):
    print(len(data))
    coefficients = pywt.wavedec(data.to_numpy().reshape((50,)), mother_wavelet, level=n_levels, mode='per')
    sigma = mad(coefficients[-1])
    thresh = sigma * np.sqrt(2 * np.log(len(data)))
    coefficients[1:] = (pywt.threshold(i, value=thresh, mode='hard') for i in coefficients[1:])
    clean = pywt.waverec(coefficients, mother_wavelet, mode='per')
    plt.plot(data, c='g')
    plt.plot(clean, c='r')
    plt.show()

'''
