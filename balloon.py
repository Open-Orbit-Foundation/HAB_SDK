import math

class Balloon:
    balloons = 0
    heliumMolarMass = 4.002602
    standardTemperature = 273.15
    standardPressure = 101.325
    gasConstant = 1.380622 * 6.022169

    def __init__(self, mass, gas, diameter, dragCoefficient, parachuteDragCoefficient, parachuteCrossSection):
        #Balloon.balloons += 1
        self.mass = mass
        self.gas = gas
        self.diameter = diameter
        self.dragCoefficient = dragCoefficient
        self.parachuteDragCoefficient = parachuteDragCoefficient
        self.parachuteCrossSection = parachuteCrossSection

    def burstVolume(self):
        burstVolume = 4 / 3 * math.pi * (self.diameter / 2) ** 3
        return burstVolume
        
    def totalMass(self):
        totalMass = self.mass + self.gas * Balloon.heliumMolarMass * Balloon.standardPressure / (Balloon.gasConstant * Balloon.standardTemperature)
        return totalMass
    
    def Volume(self, pressure):
        volume = self.gas * Balloon.standardPressure / pressure
        return volume
    
    def crossSection(self, volume):
        area = math.pi * (volume * 3 / (4 * math.pi)) ** (2 / 3)
        return area