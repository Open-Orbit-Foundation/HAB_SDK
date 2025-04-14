import numpy as np
import time
#import math
import bisect
import matplotlib.pyplot as plt
#import matplotlib.ticker as tic
from balloon import Balloon
from atmosphere import standardAtmosphere

launchAltitude = 0

balloon1 = Balloon(0.6 + 0.6, 1.2, 19.8 * 0.3048, 0.55)
atmosphere = standardAtmosphere()

def calculateAscentRate(dragCoefficient, gravity, density, volume, mass, crossSection):
    ascentRate = (2 * gravity * (density * volume - mass) / (dragCoefficient * density * crossSection)) ** 0.5
    return ascentRate

pressures = []
temperatures = []
densities = []
volumes = [balloon1.gas]
crossSections = []
ascentRates = []
altitudes = [launchAltitude]
times = [0]
timeStep = 10 #sec

while volumes[-1] <= balloon1.burstVolume():
    i = bisect.bisect(atmosphere.segments, altitudes[-1]) - 1
    pressures.append(atmosphere.Pressure(i, altitudes[-1]))
    temperatures.append(atmosphere.Temperature(i, altitudes[-1]))
    densities.append(atmosphere.Density(pressures[-1], temperatures[-1]))
    volumes.append(balloon1.Volume(pressures[-1]))
    crossSections.append(balloon1.crossSection(volumes[-1]))
    ascentRates.append(calculateAscentRate(balloon1.dragCoefficient, atmosphere._Gravity(altitudes[-1]), densities[-1], volumes[-1], balloon1.totalMass(), crossSections[-1]))
    #print("Time: " + str(times[-1]) + " | Altitude: " + str(altitudes[-1]) + " | Ascent Rate: " + str(ascentRates[-1]) + " | Volume: " + str(volumes[-1]))
    altitudes.append(altitudes[-1] + ascentRates[-1] * timeStep)
    times.append(times[-1] + timeStep / 60)

plt.plot(times[:-1], np.array(altitudes[:-1]) / 1000)
plt.xlabel("Time (min)")
plt.ylabel("Altitude (km)")
plt.title("Ascent Profile")
plt.legend()
plt.grid(True, which='both', linestyle='--')
plt.show()

'''
fig, ax1 = plt.subplots()

ax1.plot(pressures, np.array(altitudes[:-1]) / 1000, "b-", label="Pressure")
ax1.set_xlabel("Pressure (kPa)")
ax1.set_ylabel("Altitude (m)")
ax1.set_xscale('log')
ax1.yaxis.set_major_locator(tic.MultipleLocator(10))
ax1.tick_params(axis = "x", colors = "blue")
ax1.legend(loc='upper right')

ax2 = ax1.twiny()
ax2.plot(densities, np.array(altitudes[:-1]) / 1000, "r--", label="Density")
ax2.set_xlabel("Density (kg/m^3)")
ax2.set_xscale('log')
ax2.tick_params(axis = "x", colors = "red")
ax2.legend(loc='lower left')

ax1.set_xlim(min(min(pressures), min(densities)) / 10 ** 0.5, max(max(pressures), max(densities)) * 10 ** 0.5)
ax1.set_ylim(min(np.array(altitudes[:-1]) / 1000), np.ceil(max(np.array(altitudes[:-1]) / 1000) / 10) * 10)
ax2.set_xlim(ax1.get_xlim())

plt.title("ISA Profile")
plt.grid(True)
plt.tight_layout()
plt.show()
'''