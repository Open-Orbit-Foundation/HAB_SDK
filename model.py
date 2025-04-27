import numpy as np
from atmosphere import standardAtmosphere

atmosphere = standardAtmosphere()

class Model:
    timeStep = 10 #seconds

    def __init__(self, profiles, launchAltitude, result):
        self.profiles = profiles
        self.launchAltitude = launchAltitude
        self.result = result

    def altitudeModel(self):
        for profile in self.profiles:
            altitudes = [self.launchAltitude]
            times = [0]
            volume = profile.gas
            while volume <= profile.burstVolume():
                pressure, density = atmosphere._Qualities(altitudes[-1])
                volume = profile.Volume(pressure)
                ascentRate = np.sign(density * volume - profile.totalMass()) * (2 * atmosphere._Gravity(altitudes[-1]) * abs(density * volume - profile.totalMass()) / (profile.dragCoefficient * density * profile.crossSection(volume))) ** 0.5
                if ascentRate > 0:
                    altitudes.append(altitudes[-1] + ascentRate * Model.timeStep)
                else:
                    break
                times.append(times[-1] + Model.timeStep / 60)
            while altitudes[-1] > altitudes[0]:
                pressure, density = atmosphere._Qualities(altitudes[-1])
                descentRate = - ((2 * atmosphere._Gravity(altitudes[-1]) * profile.mass) / (profile.parachuteDragCoefficient * density * profile.parachuteCrossSection)) ** 0.5
                altitudes.append(altitudes[-1] + descentRate * Model.timeStep)
                times.append(times[-1] + Model.timeStep / 60)
            if altitudes[-1] <= altitudes[0]:
                del times[-1]
                del altitudes[-1]
            self.result.append((times[:-1], altitudes[:-1]))
        return self.result