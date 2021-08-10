import yt

from yt.extensions.astro_analysis.halo_analysis.halo_catalog.halo_callbacks import \
    periodic_distance

from yt.utilities.exceptions import \
    YTSphereTooSmall

def dirname(path, up=0):
    return "/".join(path.split('/')[:-up-1])

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
        try:
            sphere = sphere.ds.sphere(com, (1-stepsize) * sphere.radius)
            yield sphere
        except YTSphereTooSmall:
            yield None
            break

def sphere_icom(sphere, inner_radius, stepsize=0.1,
                com_kwargs=None, verbose=True):
    center_orig = sphere.center
    old_center = center_orig

    for new_sphere in iterate_center_of_mass(
            sphere, inner_radius, stepsize, com_kwargs):
        if new_sphere is None:
            break

        new_center = new_sphere.center
        if verbose:
            diff = uperiodic_distance(
                old_center.to("unitary"),
                new_center.to("unitary"))
            yt.mylog.info(
                    "Radius: %s, center: %s, diff: %s." %
                    (new_sphere.radius.to('pc'),
                     new_sphere.center.to('unitary'),
                     diff.to('pc')))

        sphere = new_sphere
        old_center = new_center

    if verbose:
        distance = uperiodic_distance(
            center_orig.to("unitary"),
            new_center.to("unitary"))
        yt.mylog.info("Recentering sphere %s away." %
                       distance.to("pc"))

    return sphere.center

def reunit(ds, val, units):
    if isinstance(val, yt.YTQuantity):
        func = ds.quan
    else:
        func = ds.arr
    return func(val.to(units).d, units)

def uperiodic_distance(x1, x2, domain=None):
    units = x1.units
    x2.convert_to_units(units)
    if domain is None:
        dom = None
    else:
        dom = domain.to(units).v

    d = periodic_distance(x1.v, x2.v, dom)
    return d * x1.uq
