import bisect
import math

class standardAtmosphere:
    """
    1976 US Standard Atmosphere (geopotential formulation).

    Inputs:
      - Zh : geometric altitude [m]

    Outputs:
      - p  : pressure [kPa]
      - T  : temperature [K]
      - rho: density [kg/m^3]
      - g  : gravitational acceleration [m/s^2]
    """
    _segments = [0, 11, 20, 32, 47, 51, 71, 84.852] #geopotential altitude segments in km
    _lapseRates = [-6.5, 0, 1, 2.8, 0, -2.8, -2, 0] #lapse rate @ each segment altitude
    _temperatures = [288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.946] #temperature @ each segment altitude
    _pressures = [101.325, 22.632142, 5.474889, 0.868019, 0.110906, 0.066939, 0.003956, 0.000373] #pressure @ each segment altitude

    def _Qualities(self, Zh):
        Zh = max(Zh, 0)
        Z = Zh/1000
        g = 9.80665 * (6356.766 / (6356.766 + Z)) ** 2
        H = 6356.766 * Z / (6356.766 + Z)
        i = bisect.bisect(self._segments, H) - 1
        t = self._temperatures[i] + self._lapseRates[i] * (H - self._segments[i])
        if self._lapseRates[i] != 0:
            p = self._pressures[i] * (self._temperatures[i] / t) ** (34.163195 / self._lapseRates[i])
        else:
            p = self._pressures[i] * math.exp(-34.163195 * (H - self._segments[i]) / self._temperatures[i])
        rho = p * 3.483676 / t
        return p, t, rho, g