import numpy as np



# This will generate a profile for a 1D soliton

def create_profile(num_grid_points: int) -> np.array:
    """
    Creates a profile function for a grid with 
    `num_grid_points` points.

    Parameters
    ----------
    num_grid_points: int
        Number of grid points

    Returns
    -------
    profile: np.array
        Generated profile function 
    """

    profile = np.zeros(num_grid_points)
    return profile




# Define a class called Soliton

class Soliton():
    # Define the initialisation method, which will accept two numbers, the number of grid points and the grid spacing
    def __init__(self, num_grid_points, grid_spacing):
         self.lp = num_grid_points
         self.ls = grid_spacing
         self.profile = create_profile(num_grid_points)

    # Define the method to compute the energy
    def compute_energy(self):
         total_energy = np.sum(self.profile)
         total_energy *= self.ls
         self.energy = total_energy
    # Now we have a method that takes in the profile function, sums it up, multiplies this value by the grid spacing, stores the value in self.energy then returns this value.


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