def calculate_box_mass(x, y, z, density=300):
    '''
    Calculates mass for the phalanges with rectangular shape
    Parameters:
        x, y, z: the size attributes
        density: light wood with density of 300kg/m3
                source: https://www.britannica.com/science/wood-plant-tissue/Properties-of-wood
    Returns:
        volume * density
    '''
    return round((x * y * z) * density, 3)


def moments_of_inertia_box(x, y, z, mass):
    '''
    Calculates moments of inertia for the phalanges with rectangular shape
    Parameters:
        x, y, z: the size attributes
        mass: the mass of the shape
    Source: https://en.wikipedia.org/wiki/List_of_moments_of_inertia
    '''
    ixx = (1 / 12) * mass * (y**2 + z**2)
    iyy = (1 / 12) * mass * (x**2 + z**2)
    izz = (1 / 12) * mass * (x**2 + y**2)

    return round(ixx, 3), round(iyy, 3), round(izz, 3)


def normalize(val, t_min, t_max, r_min=0, r_max=1):
    '''
    Normalizes a value from an original range to a target range.
    In this case, r_min and r_max are 0 and 1 respectively because the
    genome matrix contains random floats between 0 and 1
    Params:
        val: value
        t_min: target range minimum
        t_max: target range maximum
        r_min: original range minimum
        r_max: original range maximum
    '''
    return ((t_max - t_min) / (r_max - r_min) * (val - r_min)) + t_min
