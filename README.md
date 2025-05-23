# EXP.NO.8-Simulation-of-QPSK

8.Simulation of QPSK

# AIM
To simulate the quadrature phase shift keying modulation using python and visualize in phase, quadrature and resulatant waveforms.

# SOFTWARE REQUIRED
google colab

# ALGORITHMS
1. Import required libraries.

2. Define simulation parameters (symbol period, sampling frequency, etc.).

3. Generate a random bit sequence.

4. Map bits into QPSK symbols (2 bits per symbol).

5. Assign each symbol a phase from the QPSK constellation.

6. Generate QPSK-modulated waveform using cosine and sine for I/Q components.


# PROGRAM
 Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

 Parameters
num_symbols = 10             # Number of QPSK symbols
T = 1.0                      # Symbol period (seconds)
fs = 100.0                   # Sampling frequency (Hz)
t = np.arange(0, T, 1/fs)    # Time vector for one symbol

 Generate random bit sequence
bits = np.random.randint(0, 2, num_symbols * 2)  # Two bits per symbol
symbols = 2 * bits[0::2] + bits[1::2]            # Map bits to symbol (0 to 3)

 Initialize QPSK signal
qpsk_signal = np.array([])
symbol_times = []

 Define phase mapping for QPSK (Gray Coding)
symbol_phases = {
    0: 0,
    1: np.pi / 2,
    2: np.pi,
    3: 3 * np.pi / 2
}

 Generate the QPSK modulated signal
for i, symbol in enumerate(symbols):
    phase = symbol_phases[symbol]
    symbol_time = i * T
    qpsk_segment = np.cos(2 * np.pi * t / T + phase) + 1j * np.sin(2 * np.pi * t / T + phase)
    qpsk_signal = np.concatenate((qpsk_signal, qpsk_segment))
    symbol_times.append(symbol_time)

 Full time vector
t_total = np.arange(0, num_symbols * T, 1/fs)

 Plotting the QPSK signal
plt.figure(figsize=(14, 12))

In-phase component
plt.subplot(3, 1, 1)
plt.plot(t_total, np.real(qpsk_signal), label='In-phase (I)', color='blue')
for i, symbol_time in enumerate(symbol_times):
    plt.axvline(symbol_time, color='red', linestyle='--', linewidth=0.5)
    plt.text(symbol_time + T/4, 0, f'{symbols[i]:02b}', fontsize=12, color='black')
plt.title('QPSK - In-phase Component')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

 Quadrature component
plt.subplot(3, 1, 2)
plt.plot(t_total, np.imag(qpsk_signal), label='Quadrature (Q)', color='orange')
for i, symbol_time in enumerate(symbol_times):
    plt.axvline(symbol_time, color='red', linestyle='--', linewidth=0.5)
    plt.text(symbol_time + T/4, 0, f'{symbols[i]:02b}', fontsize=12, color='black')
plt.title('QPSK - Quadrature Component')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

 Resultant QPSK waveform
plt.subplot(3, 1, 3)
plt.plot(t_total, np.real(qpsk_signal), label='Resultant QPSK Waveform', color='green')
for i, symbol_time in enumerate(symbol_times):
    plt.axvline(symbol_time, color='red', linestyle='--', linewidth=0.5)
    plt.text(symbol_time + T/4, 0, f'{symbols[i]:02b}', fontsize=12, color='black')
plt.title('Resultant QPSK Waveform (Real Part)')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
# OUTPUT
![Screenshot 2025-05-22 155744](https://github.com/user-attachments/assets/9f8eaf83-1b9d-4e33-9337-7fe8ad895038)


 
# RESULT / CONCLUSIONS
The simulation successfully generated the QPSK modulated waveform. The in-phase and quadrature components were plotted, along with the resultant QPSK waveform. The symbol transitions and binary mapping are clearly visualized.
