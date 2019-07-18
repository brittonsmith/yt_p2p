import yt

def iterate_center_of_mass(sphere, inner_radius, stepsize=0.05,
                           com_kwargs=None):
    """
    Return a series of spheres centered on the current center of
    mass with radius decreasing by stepsize. Stop when inner_radius
    is reached.
    """

    if com_kwargs is None:
        com_kwargs = {}

    yield sphere
    while (sphere.radius > inner_radius):
        com = sphere.quantities.center_of_mass(**com_kwargs)
        sphere = sphere.ds.sphere(com, (1-stepsize) * sphere.radius)
        yield sphere

def reunit(ds, val, units):
    if isinstance(val, yt.YTQuantity):
        func = ds.quan
    else:
        func = ds.arr
    return func(val.to(units).d, units)
