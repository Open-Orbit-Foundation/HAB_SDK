import numpy as np
import math
import bisect
import matplotlib.pyplot as plt
import matplotlib.ticker as tic

burstAltitude = 30000
launchAltitude = 0

balloonMass = 0.6 #kg
payloadMass = 0.6 #kg
gasVolume = 1.2 #m^3
gasMolarMass = 4.002602
standardTemperature = 273.15 #K
dragCoefficient = 0.55
burstDiameter = 19.8 * 0.3048 #m
burstVolume = 4 / 3 * math.pi * (burstDiameter / 2) ** 3

g0 = 9.80665
R = 1.380622 * 6.022169
M0 = 28.9644
P0 = 101.325 #kPa
r0 = 6356766 #meters

gasMass = gasVolume * gasMolarMass * P0 / (R * standardTemperature)

segments = [0, 11000, 20000, 32000, 47000, 51000, 71000, 84852] #geopotentialAltitude
lapseRates = [-6.5, 0, 1, 2.8, 0, -2.8, -2, 0] #K/km

def calculateGraviationalAcceleration(geometricAltitude): #meters
    graviationalAcceleration = g0 * (r0 / (r0 + geometricAltitude)) ** 2
    return graviationalAcceleration

def calculateGeopotentialAltitude(geometricAltitude): #meters
    geopotentialAltitude = geometricAltitude * r0 / (geometricAltitude + r0)
    return geopotentialAltitude

def calculateVolume(initialVolume, initialPressure, pressure):
    volume = initialVolume * initialPressure / pressure
    return volume

def estimateBalloonCrossSection(volume):
    balloonCrossSection = math.pi * (volume * 3 / (4 * math.pi)) ** (2 / 3)
    return balloonCrossSection

def calculateAscentRate(dragCoefficient, gravity, density, volume, mass, crossSection):
    ascentRate = (2 * gravity * (density * volume - mass) / (dragCoefficient * density * crossSection)) ** 0.5
    return ascentRate

baseTemperatures = [288.15] #K

def calculateTemperature(index, geopotentialAltitude): #int, meters
    temperature = baseTemperatures[i] + lapseRates[i] * (geopotentialAltitude - segments[i]) / 1000
    return temperature

for i in range(len(segments) - 1):
    baseTemperatures.append(calculateTemperature(i, segments[i + 1]))

basePressures = [P0]

def calculatePressure(index, geopotentialAltitude): #int, meters
    if lapseRates[i] != 0:
        pressure = basePressures[i] * (baseTemperatures[i] / ((baseTemperatures[i] + lapseRates[i] * (geopotentialAltitude - segments[i]) / 1000))) ** (g0 * M0 / (R * lapseRates[i]))
    else:
        pressure = basePressures[i] * np.exp((-g0 * M0 * (geopotentialAltitude - segments[i]) / 1000) / (R * baseTemperatures[i]))
    return pressure

for i in range(len(segments) - 1):
    basePressures.append(calculatePressure(i, segments[i + 1]))

def calculateDensity(index, pressure, temperature):
    density = pressure * M0 / (R * temperature)
    return density

pressures = []
temperatures = []
densities = []
volumes = [gasVolume]
crossSections = []
ascentRates = []
altitudes = [launchAltitude]
times = [0]
timeStep = 10 #sec

while volumes[-1] <= burstVolume:
    i = bisect.bisect(segments, altitudes[-1]) - 1
    pressures.append(calculatePressure(i, calculateGeopotentialAltitude(altitudes[-1])))
    temperatures.append(calculateTemperature(i, calculateGeopotentialAltitude(altitudes[-1])))
    densities.append(calculateDensity(i, pressures[-1], temperatures[-1]))
    volumes.append(calculateVolume(gasVolume, P0, pressures[-1]))
    crossSections.append(estimateBalloonCrossSection(volumes[-1]))
    ascentRates.append(calculateAscentRate(dragCoefficient, calculateGraviationalAcceleration(altitudes[-1]), densities[-1], volumes[-1], balloonMass + payloadMass + gasMass, crossSections[-1]))
    print("Time: " + str(times[-1]) + " | Altitude: " + str(altitudes[-1]) + " | Ascent Rate: " + str(ascentRates[-1]) + " | Volume: " + str(volumes[-1]))
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