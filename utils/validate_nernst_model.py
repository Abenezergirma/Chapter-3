import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['text.usetex'] = True

x = np.linspace(0, 10, 100)
y = np.sin(x)

fig, ax = plt.subplots()

ax.plot(x, y, label=r'\textsf{\textbf{Sine Curve}}')  # Bold sans-serif in legend
ax.set_title(r'\textit{Voltage Profile}', fontsize=14)  # Italic title
ax.set_xlabel(r'\textrm{Time (s)}', fontsize=12)       # Roman font for x-axis
ax.set_ylabel(r'\texttt{Voltage (V)}', fontsize=12)    # Typewriter font for y-axis
ax.legend()

plt.show()
