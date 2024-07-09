from matplotlib import pyplot
import numpy as np
import yaml

from grid_figure import GridFigure

pyplot.rcParams['font.size'] = 16

# Taken from Cloudy documentation.
solar_abundance = {
    'H' : 1.00e+00, 'He': 1.00e-01, 'Li': 2.04e-09,
    'Be': 2.63e-11, 'B' : 6.17e-10, 'C' : 2.45e-04,
    'N' : 8.51e-05, 'O' : 4.90e-04, 'F' : 3.02e-08,
    'Ne': 1.00e-04, 'Na': 2.14e-06, 'Mg': 3.47e-05,
    'Al': 2.95e-06, 'Si': 3.47e-05, 'P' : 3.20e-07,
    'S' : 1.84e-05, 'Cl': 1.91e-07, 'Ar': 2.51e-06,
    'K' : 1.32e-07, 'Ca': 2.29e-06, 'Sc': 1.48e-09,
    'Ti': 1.05e-07, 'V' : 1.00e-08, 'Cr': 4.68e-07,
    'Mn': 2.88e-07, 'Fe': 2.82e-05, 'Co': 8.32e-08,
    'Ni': 1.78e-06, 'Cu': 1.62e-08, 'Zn': 3.98e-08}

atomic_mass = {
    'H' : 1.00794,   'He': 4.002602,  'Li': 6.941,
    'Be': 9.012182,  'B' : 10.811,    'C' : 12.0107,
    'N' : 14.0067,   'O' : 15.9994,   'F' : 18.9984032,
    'Ne': 20.1797,   'Na': 22.989770, 'Mg': 24.3050,
    'Al': 26.981538, 'Si': 28.0855,   'P' : 30.973761,
    'S' : 32.065,    'Cl': 35.453,    'Ar': 39.948,
    'K' : 39.0983,   'Ca': 40.078,    'Sc': 44.955910,
    'Ti': 47.867,    'V' : 50.9415,   'Cr': 51.9961,
    'Mn': 54.938049, 'Fe': 55.845,    'Co': 58.933200,
    'Ni': 58.6934,   'Cu': 63.546,    'Zn': 65.409}

atomic_number = {
    'H' : 1,  'He': 2,  'Li': 3,
    'Be': 4,  'B' : 5,  'C' : 6,
    'N' : 7,  'O' : 8,  'F' : 9,
    'Ne': 10, 'Na': 11, 'Mg': 12,
    'Al': 13, 'Si': 14, 'P' : 15,
    'S' : 16, 'Cl': 17, 'Ar': 18,
    'K' : 19, 'Ca': 20, 'Sc': 21,
    'Ti': 22, 'V' : 23, 'Cr': 24,
    'Mn': 25, 'Fe': 26, 'Co': 27,
    'Ni': 28, 'Cu': 29, 'Zn': 30}

def Dtrans(aC, aO):
    return np.log10(aC/solar_abundance["C"] + 0.3 * aO/solar_abundance["O"])

def Dtrans_logsol(aC, aO):
    return np.log10(10**aC + 0.3 * 10**aO)

def read_table(fn):
    lines = open(fn, mode="r").readlines()
    header = lines.pop(0).strip()
    keys = header.split(",")[1:]

    data = {}
    for line in lines:
        vals = line.strip().split(",")
        jid = vals.pop(0)

        datum = {k:v for k, v in zip(keys, vals)}
        data[jid] = datum

    return data

def enumerize(data, keys):
    for star, datum in data.items():
        for key in keys:
            val = datum[key]
            datum[f"{key}_upper"] = val.startswith("<")
            datum[key] = float(val[1:])

def combine_datums(dicts):
    data = {}
    all_keys = set(list(dataO.keys()) + list(dataC.keys()))
    for key in all_keys:
        datum = {}
        for my_dict in dicts:
            datum.update(my_dict.get(key, {}))

        data[key] = datum
    return data

def get_model_Zcr(fn, tol, method):
    with open(fn, mode="r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    values = []
    for star, datum in data.items():
        row = []
        value = datum["solutions"][tol][f"evaluate_model_{method}"]["value"]
        row.append(f"{value:.2f}")
        values.append(value)

    return np.mean(values), np.std(values)

if __name__ == "__main__":
    dataC = read_table("table_C_Fe.csv")
    dataO = read_table("table_O_Fe.csv")

    enumerize(dataC, ["Fe/H", "C/H"])
    enumerize(dataO, ["Fe/H", "O/H"])

    data = combine_datums([dataC, dataO])

    my_fig = GridFigure(1, 1, figsize=(8, 6),
                        top_buffer=0.02, right_buffer=0.02,
                        left_buffer=0.09, bottom_buffer=0.1)
    my_axes = my_fig[0]
    cmap = pyplot.cm.turbo

    c1 = cmap(0.05)
    sel = [d for d in data.values() if 'C/H' in d and 'O/H' in d]
    feh1 = [d["Fe/H"] for d in sel]
    ch1 = np.array([d["C/H"] for d in sel])
    oh1 = np.array([d["O/H"] for d in sel])
    dt1 = np.log10(10**ch1 + 0.3 * 10**oh1)
    xup = np.array([d["Fe/H_upper"] for d in sel])
    yup = np.array([d["C/H_upper"] or d["O/H_upper"] for d in sel])
    my_axes.errorbar(feh1, dt1, fmt='o', markersize=3, label="C+O",
                     alpha=0.5, color=c1,
                     xerr=0.1*xup, yerr=0.1*yup,
                     xuplims=xup, uplims=yup)
    print (f"C/O combined, Dtrans,min = {dt1.min()} at [Fe/H] = {feh1[dt1.argmin()]}.")

    c2 = cmap(0.7)
    sel = [d for d in data.values() if 'C/H' in d and 'O/H' not in d]
    feh2 = [d["Fe/H"] for d in sel]
    ch2 = np.array([d["C/H"] for d in sel])
    dt2 = np.log10(10**ch2)
    xup = np.array([d["Fe/H_upper"] for d in sel])
    yup = np.array([d["C/H_upper"] for d in sel])
    my_axes.errorbar(feh2, dt2, fmt='o', markersize=3, label="C only",
                     alpha=0.5, color=c2,
                     xerr=0.1*xup, yerr=0.1*yup,
                     xuplims=xup, uplims=yup)
    print (f"C only, Dtrans,min = {dt2.min()} at [Fe/H] = {feh2[dt2.argmin()]}.")

    c3 = cmap(0.9)
    sel = [d for d in data.values() if 'C/H' not in d and 'O/H' in d]
    feh3 = [d["Fe/H"] for d in sel]
    oh3 = np.array([d["O/H"] for d in sel])
    dt3 = np.log10(0.3 * 10**oh3)
    xup = np.array([d["Fe/H_upper"] for d in sel])
    yup = np.array([d["O/H_upper"] for d in sel])
    my_axes.errorbar(feh3, dt3, fmt='o', markersize=3, label="O only",
                     alpha=0.5, color=c3,
                     xerr=0.1*xup, yerr=0.1*yup,
                     xuplims=xup, uplims=yup)
    print (f"O only, Dtrans,min = {dt3.min()} at [Fe/H] = {feh3[dt3.argmin()]}.")

    my_axes.xaxis.set_label_text("[Fe/H]")
    my_axes.yaxis.set_label_text("$D_{\\rm trans}$")
    pyplot.legend(loc="best")

    Zcr_m, Zcr_s = get_model_Zcr("../models.yaml", "0.001", "collapsed")
    
    Dcr = Zcr_m + Dtrans(solar_abundance["C"], solar_abundance["O"])
    Dcr_C = Zcr_m + Dtrans(solar_abundance["C"], 0)
    Dcr_O = Zcr_m + Dtrans(0, solar_abundance["O"])

    my_axes.axhline(y=Dcr, color=c1, alpha=0.8, linestyle="--", linewidth=3)
    my_axes.fill_between(x=[-8, -1], color=c1, alpha=0.3,
                         linewidth=0,
                         y1=Dcr-Zcr_s, y2=Dcr+Zcr_s)
    my_axes.axhline(y=Dcr_C, color=c2, alpha=0.8, linestyle="--", linewidth=3)
    my_axes.axhline(y=Dcr_O, color=c3, alpha=0.8, linestyle="--", linewidth=3)

    my_axes.tick_params(axis="x", direction="inout", which="both",
                        top=True, bottom=True)
    my_axes.tick_params(axis="y", direction="inout", which="both",
                        left=True, right=True)
    my_axes.xaxis.set_ticks(np.linspace(-7.5, -2, 12), minor=True)
    my_axes.yaxis.set_ticks(np.linspace(-4.5, 0.5, 11), minor=True)

    my_axes.grid(visible=True, axis="both", zorder=0, linestyle=":",
                 color="black", alpha=0.6)
    
    my_axes.set_xlim(-7.5, -1.75)
    
    pyplot.savefig("Dtrans.pdf")
