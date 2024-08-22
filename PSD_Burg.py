# Author: Simone Barbarino
# Mail:   simone.barbarino@proton.me

import numpy as np
from spectrum import arburg, pburg
import plotly.graph_objects as go

def PSD_Burg(signal: np.ndarray, N: int, fs: float, NFFT: int, perc: float, gr: int = 0):
    """
    Calculates the Power Spectral Density (PSD) using the Burg method.

    Args:
        signal (np.ndarray): The input signal for which the PSD is to be calculated.
        N (int): The maximum order of the AutoRegressive (AR) model.
        fs (float): The sampling frequency of the signal.
        NFFT (int): The number of points used in the FFT to calculate the PSD.
        perc (float): Percentage used to determine the AR model order using the variance method.
        gr (int): If non-zero, enables plotting of the PSD (default is 0, meaning no plotting).

    Returns:
        Pb (np.ndarray): The calculated Power Spectral Density values.
        fb (np.ndarray): The corresponding frequency values for the PSD.
    """
    
    # Remove the mean from the signal to center it around zero
    signal = signal - np.mean(signal)

    # Initialize an array to store the prediction errors for different AR model orders
    e1 = np.zeros(N + 1)
    for NN in range(2, N + 1):
        # Calculate the AR model and get the prediction error for each order
        _, e1[NN], _ = arburg(signal, NN)
    
    # Calculate a threshold value to determine the optimal AR model order
    asint1 = perc * e1[-1] / 100 + e1[-1]
    ind1 = np.where(e1[2:N + 1] < asint1)[0]

    # Select the optimal AR model order based on the calculated threshold
    if ind1.size > 0:
        order1 = ind1[0] + 1  # Adjust for zero-based index
    else:
        order1 = N  # Use the maximum order if no lower order is found

    # Compute the PSD using the Burg method with the selected AR order
    p = pburg(signal, order1, NFFT=NFFT, sampling=fs)
    Pb = p.psd
    fb = np.array(p.frequencies())

    # If plotting is enabled (gr != 0), generate the plot of the PSD
    if gr:
        # Create the plot
        fig = go.Figure()

        # Add the PSD trace to the plot
        fig.add_trace(go.Scatter(
            x=fb,
            y=[10 * np.log10(p) for p in Pb],  # Convert PSD to dB scale
            mode='lines+markers',
            marker=dict(symbol='circle', size=8),
            line=dict(shape='linear')
        ))

        # Set the plot's title and axis labels
        fig.update_layout(
            title='Power Spectral Density using Burg Method',
            xaxis_title='Frequency (Hz)',
            yaxis_title='Power Spectral Density (dB/Hz)',
            template='plotly_white'
        )

        # Display the plot
        fig.show()

    # Return the PSD values and corresponding frequencies
    return Pb, fb
