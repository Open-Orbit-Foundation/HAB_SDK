import math

class Integrator:
    @staticmethod
    def rk4_second_order(x, dx, ddx_func, dt):
        """
        Classical RK4 integrator for second-order systems.

        This is used for both scalar and vector states:
          - x   : position vector (or scalar)
          - dx  : velocity vector (or scalar)
          - ddx : acceleration as a function of (x, dx)

        Returns the post-step acceleration so it can be recorded
        without an additional force evaluation.
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

class Utility:
    @staticmethod
    def progress_bar(initial, current, final, prefix="", suffix="", ascending: bool = True, bar_length=100):
        # Ensure we clamp correctly even when initial > final (e.g., descent)
        lo = min(initial, final)
        hi = max(initial, final)
        current = max(lo, min(hi, current))

        span = hi - lo  # always >= 0 now

        # Define progress value along the same axis:
        # - ascending=True means progress goes initial -> final
        # - ascending=False means progress goes final -> initial (useful when you pass "current altitude")
        if span == 0:
            fraction = 1.0
        else:
            if ascending:
                # progress from initial to current
                fraction = abs(current - initial) / span
            else:
                # progress from current to final (i.e., "how far toward final")
                fraction = abs(final - current) / span

        # Clamp fraction to [0, 1] to prevent any outliers from creating huge strings
        fraction = max(0.0, min(1.0, fraction))

        completed = int(bar_length * fraction)
        completed = max(0, min(bar_length, completed))

        bar = "#" * completed + "-" * (bar_length - completed)
        return f"\r{prefix} |{bar}| {fraction * 100:>3.0f}% {suffix}"