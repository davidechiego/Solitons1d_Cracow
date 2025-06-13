import numpy as np
from typing import Callable
import matplotlib.pyplot as plt




# Define the Grid class

# A user can create a grid by specifying the number of grid points and the grid spacing. The Grid will then compute a grid (saved as numpy array) and save the length of the grid.

class Grid:
    """
    A 1D grid.

    Parameters
    ----------
    num_grid_points : int
        Number of grid points used in grid.
    grid_spacing : float
        Spacing between grid points.

    Attributes
    ----------
    grid_length : float
        Total length of grid.
    grid_points : np.array[float]
        An array of grid points, of the grid.

    """

    def __init__(
        self,
        num_grid_points: int,
        grid_spacing: float,
    ):
        self.num_grid_points = num_grid_points
        self.grid_spacing = grid_spacing
        self.grid_length = (num_grid_points) * grid_spacing

        self.grid_points = np.arange(
            -self.grid_length / 2, self.grid_length / 2, grid_spacing
        )



# Define the Lagrangian class

# The Lagrangian will keep track of the potential function. We won’t set up automatic differentiation, so we’ll also give the Lagrangian the derivative of the potential. We’ll also optionally allow the user to input the vacua of the theory. If they do, we’ll add a check that the derivative function of the vacua return 0. This class could also be where we define how many fields our theory has, any funny metric, and more. But let’s keep it simple for now.

class Lagrangian:
    """
    Used to represent Lagrangians of the form:
        L = - 1/2(dx_phi)^2 - V(phi)

    Parameters
    ----------
    V : function
        The potential energy function, must be a map from R -> R
    dV : function
        The derivative of the potential energy function, must be a map from R -> R
    vacua : list-like or None
        List of vacua of the potential energy.
    """

    def __init__(
        self,
        V: Callable[[float], float], # Yup - you can pass functions are argument in python!
        dV: Callable[[float], float],
        vacua: list | np.ndarray | None = None,  # np.ndarray is the type of a numpy array
    ):
        self.V = V
        self.dV = dV
        self.vacua = vacua

        if vacua is not None:
            for vacuum in vacua:
                # np.isclose does what it sounds like: are the values close?
                # That f"" is called an f-string, allowing you to add parameters to strings
                assert np.isclose(dV(vacuum), 0), (
                    f"The given vacua do not satisfy dV({vacuum}) = 0"
                )


# Soliton class
class Soliton:
    """
    A class describing a Soliton.

    Parameters
    ----------
    grid : Grid
        The grid underpinning the soliton.
    lagrangian : Lagrangian
        The Lagrangian of the theory supporting the soliton.
    initial_profile_function : None | function
        The initial profile function, must be from R -> R. Optional.
    initial_profile : None | array-like
        The initial profile function as an array. Optional.
    """

    def __init__(
        self,
        grid: Grid,
        lagrangian: Lagrangian,
        initial_profile_function: Callable[[float], float] | None = None,
        initial_profile: np.ndarray | None = None,
    ):
        self.grid = grid
        self.lagrangian = lagrangian

        self.profile = np.zeros(grid.num_grid_points)

        assert (initial_profile_function is None) or (initial_profile is None), (
            "Please only specify `initial_profile_function` or `profile_function`"
        )

        if initial_profile_function is not None:
            self.profile = create_profile(self.grid.grid_points, initial_profile_function)
        else:
            self.profile = initial_profile

        self.energy = self.compute_energy()

    def compute_energy(self):
        """Computes the energy of a soliton, and stores this in `Soliton.energy`."""

        energy = compute_energy_fast(
            self.lagrangian.V,
            self.profile,
            self.grid.num_grid_points,
            self.grid.grid_spacing,
        )
        self.energy = energy

    def plot_soliton(self):
        """Makes a plot of the profile function of your soliton"""

        fig, ax = plt.subplots()
        ax.plot(self.grid.grid_points, self.profile)
        ax.set_title(f"Profile function. Energy = {self.energy:.4f}")

        return fig
        


# Energy fast function
def compute_energy_fast(V, profile, num_grid_points, grid_spacing):

    total_energy = 0
    return total_energy

# Profile function
def create_profile(
    grid_points: np.array,
    initial_profile_function: Callable[[np.array], np.array] | None = None,
) -> np.array:
    """
    Creates a profile function on a grid, from profile function `initial_profile_function`.

    Parameters
    ----------
    grid_points: Grid
        The x-values of a grid.
    initial_profile_function: function
        A function which accepts and returns a 1D numpy array

    Returns
    -------
    profile: np.array
        Generated profile function
    """

    profile = initial_profile_function(grid_points)
    return profile


# Compute derivatives

#So, we’ll write functions to compute the first and (while we’re at it) second derivatives of a function. (This is going to be a FAST function, so we don’t want it to interact with the class):

# First derivative
def get_first_derivative(
    phi: np.ndarray, 
    num_grid_points: int, 
    grid_spacing: float,
) -> np.ndarray:
    """
    For a given array, computes the first derivative of that array.

    Parameters
    ----------
    phi: np.ndarray
        Array to get the first derivative of
    num_grid_points: int
        Length of the array
    grid_spacing: float
        Grid spacing of underlying grid

    Returns
    -------
    d_phi: np.ndarray
        The first derivative of `phi`.

    """
    d_phi = np.zeros(num_grid_points)
    for i in np.arange(num_grid_points)[2:-2]:
        d_phi[i] = (phi[i - 2] - 8 * phi[i - 1] + 8 * phi[i + 1] - phi[i + 2]) / (
            12.0 * grid_spacing
        )

    return d_phi

# Second derivative
def get_second_derivative(
    phi: np.ndarray, 
    num_grid_points: int, 
    grid_spacing: float,
) -> np.ndarray:
    """
    For a given array, computes the first derivative of that array.

    Parameters
    ----------
    phi: np.ndarray
        Array to get the first derivative of
    num_grid_points: int
        Length of the array
    grid_spacing: float
        Grid spacing of underlying grid

    Returns
    -------
    d_phi: np.ndarray
        The first derivative of `phi`.

    """
    ddV = np.zeros(num_grid_points)
    for i in np.arange(num_grid_points)[2:-2]:
        ddV[i] = (
            -phi[i - 2] + 16 * phi[i - 1] - 30 * phi[i] + 16 * phi[i + 1] - phi[i + 2]
        ) / (12.0 * np.pow(grid_spacing, 2))

    return ddV

# Compute the total energy
def compute_energy_fast(
    V: Callable[[float], float],
    profile: np.array, 
    num_grid_points: int, 
    grid_spacing: float,
) -> float:
    """
    Computes the energy of a Lagrangian of the form
        E = 1/2 (d_phi)^2 + V(phi)

    Parameters
    ----------
    V: function
        The potential energy function
    profile: np.ndarray
        The profile function of the soliton
    num_grid_points: int
        Length of `profile`
    grid_spacing: float
        Grid spacing of underlying grid
    """
    dx_profile = get_first_derivative(profile, num_grid_points, grid_spacing)

    kin_eng = 0.5 * np.pow(dx_profile, 2)
    pot_eng = V(profile)

    tot_eng = np.sum(kin_eng + pot_eng) * grid_spacing

    return tot_eng
