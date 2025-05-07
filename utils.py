import math
import sys

class Integrator:
    @staticmethod
    def rk4_second_order(x, dx, ddx_func, dt):
        """
        RK4 integrator for second-order ODEs: d²x/dt² = f(x, dx)
        Returns: x_new, dx_new, ddx_new
        """
        k1_x = dx
        k1_dx = ddx_func(x, dx)

        k2_x = dx + 0.5 * dt * k1_dx
        k2_dx = ddx_func(x + 0.5 * dt * k1_x, dx + 0.5 * dt * k1_dx)

        k3_x = dx + 0.5 * dt * k2_dx
        k3_dx = ddx_func(x + 0.5 * dt * k2_x, dx + 0.5 * dt * k2_dx)

        k4_x = dx + dt * k3_dx
        k4_dx = ddx_func(x + dt * k3_x, dx + dt * k3_dx)

        x_new = x + (dt / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        dx_new = dx + (dt / 6.0) * (k1_dx + 2 * k2_dx + 2 * k3_dx + k4_dx)
        ddx_new = ddx_func(x_new, dx_new)

        return x_new, dx_new, ddx_new
    
    @staticmethod
    def rk2_second_order(x, dx, ddx_func, dt):
        """
        Midpoint (RK2) integrator for second-order ODEs: d²x/dt² = f(x, dx)
        Returns: x_new, dx_new, ddx_new
        """
        k1_x = dx
        k1_dx = ddx_func(x, dx)

        k2_x = dx + 0.5 * dt * k1_dx
        k2_dx = ddx_func(x + 0.5 * dt * k1_x, dx + 0.5 * dt * k1_dx)

        x_new = x + dt * k2_x
        dx_new = dx + dt * k2_dx
        ddx_new = ddx_func(x_new, dx_new)

        return x_new, dx_new, ddx_new
    
    @staticmethod
    def rk1_second_order(x, dx, ddx_func, dt):
        """
        Eulerian (RK1) integrator for second-order ODEs: d²x/dt² = f(x, dx)
        Returns: x_new, dx_new, ddx_new
        """
        k1_x = dx
        k1_dx = ddx_func(x, dx)

        x_new = x + dt * k1_x
        dx_new = dx + dt * k1_dx
        ddx_new = ddx_func(x_new, dx_new)

        return x_new, dx_new, ddx_new
    
class Geometry:
    @staticmethod
    def sphere_cross_section(V):
        return math.pi * ((3 * V) / (4 * math.pi)) ** (2 / 3)
    
class Physics:
    @staticmethod
    def volume_to_moles(V): #m^3, g/mol
        return V * 1000 * 101.325 / (1.380622 * 6.022169 * 273.15 )
    
    def volume(Vi, T, P):
        return (Vi * 101.325 / 288.15) * (T / P)
    
    def acceleration(F, m):
        return F / m

class Utility:
    @staticmethod 
    def progress_bar(current, total, prefix='', suffix='', bar_length=100):
        fraction = current / total
        completed = int(bar_length * fraction)
        bar = '#' * completed + '-' * (bar_length - completed)
        sys.stdout.write(f'\r{prefix} |{bar}| {current / total * 100:.0f}% {suffix}')