import numpy as np
from math import sqrt
from random import gauss

# Computes standard deviation from Eb/No (snr) value
def get_std_dev(snr, Eb):
    No = Eb/snr
    variance = No/2
    sigma = sqrt(variance)
    return sigma

# Applies gaussian noise with standard deviation sigma on a point (x, y) which is represented by signal
def apply_noise(signal, sigma):
    new_x = gauss(signal[0], sigma)
    new_y = gauss(signal[1], sigma)
    return [new_x, new_y]

# Emulates transmitting two bits using BPSK modulation with average bit energy of Eb
def transmit_two_bits(b0, b1, Eb):
    
    # Compute the x and the y intercepts to create the constellation
    intercept = sqrt(Eb)

    # First quadrant
    if b0 == 1 and b1 == 1:
        x = intercept
        y = intercept
        return [x, y]

    # Second quadrant
    if b0 == 0 and b1 == 1:
        x = -intercept
        y = intercept
        return [x, y]

    # Third quadrant
    if b0 == 0 and b1 == 0:
        x = -intercept
        y = -intercept
        return [x, y]

    # Fourth quadrant
    if b0 == 1 and b1 == 0:
        x = intercept
        y = -intercept
        return [x, y]

# Emulates receiving two bits using BPSK demodulation
def receive_two_bits(signal):
    r0 = 0
    r1 = 0
    if signal[0] > 0:
        r0 = 1
    if signal[1] > 0:
        r1 = 1
    return [r0, r1]

# Emulates transmitting an array of bits using QPSK
def transmit(bits, Eb):
    
    extended = False    # Tracks whether the signal was extended during transmission or not

    # If the number of bits is not even, make them even by prepending a 0
    if len(bits) % 2 != 0:
        extended = True
        bits = [0] + bits
    
    # Pick two bits as a time and map them to QPSK
    tx_signals = []
    for idx in range(0, len(bits), 2):
        signal = transmit_two_bits(bits[idx], bits[idx + 1], Eb)
        tx_signals.append(signal)

    return tx_signals, extended

# Applies channel noise on transmitted QPSK signals
def apply_channel_noise(signals, snr, Eb):
    sigma = get_std_dev(snr, Eb)
    
    # Create a noisy received signal
    out_signal = []
    for signal in signals:
        noisy_signal = apply_noise(signal, sigma)
        out_signal.append(noisy_signal)
    
    return out_signal

# Receives a noisy QPSK signal
def receive(noisy_signals, extended):
    
    out_bits =[]
    for signal in noisy_signals:
        bits = receive_two_bits(signal)
        out_bits.extend(bits)

    # If the message was extended during transmission, remove the extra redundant bit
    if extended:
        out_bits = out_bits[1:]
    
    return out_bits
