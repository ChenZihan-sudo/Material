import ase
import math

from ase.io import read as ase_read
from itertools import permutations
import numpy as np
import copy


def get_replace_atomic_numbers(compound, target_atomic_numbers):
    """
    get atomic numbers replaced list from origin one to `target_atomic_numbers` with full permutations.

    (e.g, origin: [1,1,1,1,2,2], target_atomic_numbers:[3,4], result:[[3,3,3,3,4,4],[4,4,4,4,3,3]])

    (e.g, origin: [1,2,2,2,2,3,3], target_atomic_numbers:[4,5,6], result:[[4,5,5,5,5,6,6],[4,6,6,6,6,5,5],[5,4,4,4,4,6,6],[5,6,6,6,6,4,4],[6,5,5,5,5,4,4],[6,4,4,4,4,5,5]])
    """
    target_atomic_numbers = list(set(target_atomic_numbers))

    # full permutations of target_atomic_numbers
    targets = [list(i) for i in permutations(target_atomic_numbers, len(target_atomic_numbers))]

    np_atomic_numbers = compound.get_atomic_numbers()
    set_atomic_numbers = [i for i in set(np_atomic_numbers)]

    replace_atomic_numbers = []
    for j, target in enumerate(targets):
        array = np.array(np_atomic_numbers)
        for i, d in enumerate(set_atomic_numbers):
            array[array == d] = target[i]
        replace_atomic_numbers.append(array)
    return replace_atomic_numbers


def scale_compound_volume(compound, scaling_factor):
    """
    scale the compound by volume
    """
    scaling_factor = scaling_factor ** (1 / 3)
    compound.set_cell(compound.cell * scaling_factor, scale_atoms=True)


if __name__ == "__main__":
    compound = ase_read("./dataset.spglib==2.3.1/raw/CONFIG_134.poscar", format="vasp")
    scales = [0.96, 0.98, 1.00, 1.02, 1.04]
    atomic_numbers = [58, 27, 29]  # Ce Co Cu

    print("origin one,", "volume:", round(compound.get_volume(), 2), compound)

    # get scaled compounds
    scaled_compounds = []
    for i, d in enumerate(scales):
        sacled_compound = copy.deepcopy(compound)
        scale_compound_volume(sacled_compound, d)
        scaled_compounds.append(sacled_compound)
    print("num hypothesis scaled compounds:", len(scaled_compounds))

    # get atomic numbers
    hypothesis_atomic_numbers = get_replace_atomic_numbers(compound, atomic_numbers)
    print("num hypothesis atomic numbers group:", len(hypothesis_atomic_numbers))

    # get hypothesis compounds
    hypothesis_compounds = []
    for i, sacled_compound in enumerate(scaled_compounds):
        for j, hypothesis_atomic_number in enumerate(hypothesis_atomic_numbers):
            compound = copy.deepcopy(sacled_compound)
            compound.set_atomic_numbers(hypothesis_atomic_number)
            hypothesis_compounds.append(compound)

    print("total hypothesis for single origin compound: ", len(hypothesis_compounds))
    [print("volume:", round(i.get_volume(), 2), i) for i in hypothesis_compounds]


# RESULT IN HERE:
# origin one, volume: 1218.42 Atoms(symbols='Ag24As8S24', pbc=True, cell=[[0.0, 12.021513720719087, 0.0], [6.35400996, 0.0, 0.0], [0.0, -5.816304775501783, -15.951092685696398]])
# num hypothesis scaled compounds: 5
# num hypothesis atomic numbers group: 6
# total hypothesis for single origin compound:  30
# volume: 1169.68 Atoms(symbols='Cu24Co8Ce24', pbc=True, cell=[[0.0, 11.85904091590673, 0.0], [6.268134433507226, 0.0, 0.0], [0.0, -5.7376964261313885, -15.73551095209156]])
# volume: 1169.68 Atoms(symbols='Co24Cu8Ce24', pbc=True, cell=[[0.0, 11.85904091590673, 0.0], [6.268134433507226, 0.0, 0.0], [0.0, -5.7376964261313885, -15.73551095209156]])
# volume: 1169.68 Atoms(symbols='Cu24Ce8Co24', pbc=True, cell=[[0.0, 11.85904091590673, 0.0], [6.268134433507226, 0.0, 0.0], [0.0, -5.7376964261313885, -15.73551095209156]])
# volume: 1169.68 Atoms(symbols='Ce24Cu8Co24', pbc=True, cell=[[0.0, 11.85904091590673, 0.0], [6.268134433507226, 0.0, 0.0], [0.0, -5.7376964261313885, -15.73551095209156]])
# volume: 1169.68 Atoms(symbols='Co24Ce8Cu24', pbc=True, cell=[[0.0, 11.85904091590673, 0.0], [6.268134433507226, 0.0, 0.0], [0.0, -5.7376964261313885, -15.73551095209156]])
# volume: 1169.68 Atoms(symbols='Ce24Co8Cu24', pbc=True, cell=[[0.0, 11.85904091590673, 0.0], [6.268134433507226, 0.0, 0.0], [0.0, -5.7376964261313885, -15.73551095209156]])
# volume: 1194.05 Atoms(symbols='Cu24Co8Ce24', pbc=True, cell=[[0.0, 11.940829989532327, 0.0], [6.311364312914222, 0.0, 0.0], [0.0, -5.777267996780811, -15.844035146663716]])
# volume: 1194.05 Atoms(symbols='Co24Cu8Ce24', pbc=True, cell=[[0.0, 11.940829989532327, 0.0], [6.311364312914222, 0.0, 0.0], [0.0, -5.777267996780811, -15.844035146663716]])
# volume: 1194.05 Atoms(symbols='Cu24Ce8Co24', pbc=True, cell=[[0.0, 11.940829989532327, 0.0], [6.311364312914222, 0.0, 0.0], [0.0, -5.777267996780811, -15.844035146663716]])
# volume: 1194.05 Atoms(symbols='Ce24Cu8Co24', pbc=True, cell=[[0.0, 11.940829989532327, 0.0], [6.311364312914222, 0.0, 0.0], [0.0, -5.777267996780811, -15.844035146663716]])
# volume: 1194.05 Atoms(symbols='Co24Ce8Cu24', pbc=True, cell=[[0.0, 11.940829989532327, 0.0], [6.311364312914222, 0.0, 0.0], [0.0, -5.777267996780811, -15.844035146663716]])
# volume: 1194.05 Atoms(symbols='Ce24Co8Cu24', pbc=True, cell=[[0.0, 11.940829989532327, 0.0], [6.311364312914222, 0.0, 0.0], [0.0, -5.777267996780811, -15.844035146663716]])
# volume: 1218.42 Atoms(symbols='Cu24Co8Ce24', pbc=True, cell=[[0.0, 12.021513720719087, 0.0], [6.35400996, 0.0, 0.0], [0.0, -5.816304775501783, -15.951092685696398]])
# volume: 1218.42 Atoms(symbols='Co24Cu8Ce24', pbc=True, cell=[[0.0, 12.021513720719087, 0.0], [6.35400996, 0.0, 0.0], [0.0, -5.816304775501783, -15.951092685696398]])
# volume: 1218.42 Atoms(symbols='Cu24Ce8Co24', pbc=True, cell=[[0.0, 12.021513720719087, 0.0], [6.35400996, 0.0, 0.0], [0.0, -5.816304775501783, -15.951092685696398]])
# volume: 1218.42 Atoms(symbols='Ce24Cu8Co24', pbc=True, cell=[[0.0, 12.021513720719087, 0.0], [6.35400996, 0.0, 0.0], [0.0, -5.816304775501783, -15.951092685696398]])
# volume: 1218.42 Atoms(symbols='Co24Ce8Cu24', pbc=True, cell=[[0.0, 12.021513720719087, 0.0], [6.35400996, 0.0, 0.0], [0.0, -5.816304775501783, -15.951092685696398]])
# volume: 1218.42 Atoms(symbols='Ce24Co8Cu24', pbc=True, cell=[[0.0, 12.021513720719087, 0.0], [6.35400996, 0.0, 0.0], [0.0, -5.816304775501783, -15.951092685696398]])
# volume: 1242.79 Atoms(symbols='Cu24Co8Ce24', pbc=True, cell=[[0.0, 12.101128714564323, 0.0], [6.396090722507146, 0.0, 0.0], [0.0, -5.85482447274303, -16.056732139720207]])
# volume: 1242.79 Atoms(symbols='Co24Cu8Ce24', pbc=True, cell=[[0.0, 12.101128714564323, 0.0], [6.396090722507146, 0.0, 0.0], [0.0, -5.85482447274303, -16.056732139720207]])
# volume: 1242.79 Atoms(symbols='Cu24Ce8Co24', pbc=True, cell=[[0.0, 12.101128714564323, 0.0], [6.396090722507146, 0.0, 0.0], [0.0, -5.85482447274303, -16.056732139720207]])
# volume: 1242.79 Atoms(symbols='Ce24Cu8Co24', pbc=True, cell=[[0.0, 12.101128714564323, 0.0], [6.396090722507146, 0.0, 0.0], [0.0, -5.85482447274303, -16.056732139720207]])
# volume: 1242.79 Atoms(symbols='Co24Ce8Cu24', pbc=True, cell=[[0.0, 12.101128714564323, 0.0], [6.396090722507146, 0.0, 0.0], [0.0, -5.85482447274303, -16.056732139720207]])
# volume: 1242.79 Atoms(symbols='Ce24Co8Cu24', pbc=True, cell=[[0.0, 12.101128714564323, 0.0], [6.396090722507146, 0.0, 0.0], [0.0, -5.85482447274303, -16.056732139720207]])
# volume: 1267.16 Atoms(symbols='Cu24Co8Ce24', pbc=True, cell=[[0.0, 12.179709674299831, 0.0], [6.437624942941068, 0.0, 0.0], [0.0, -5.892843878783837, -16.16099955572055]])
# volume: 1267.16 Atoms(symbols='Co24Cu8Ce24', pbc=True, cell=[[0.0, 12.179709674299831, 0.0], [6.437624942941068, 0.0, 0.0], [0.0, -5.892843878783837, -16.16099955572055]])
# volume: 1267.16 Atoms(symbols='Cu24Ce8Co24', pbc=True, cell=[[0.0, 12.179709674299831, 0.0], [6.437624942941068, 0.0, 0.0], [0.0, -5.892843878783837, -16.16099955572055]])
# volume: 1267.16 Atoms(symbols='Ce24Cu8Co24', pbc=True, cell=[[0.0, 12.179709674299831, 0.0], [6.437624942941068, 0.0, 0.0], [0.0, -5.892843878783837, -16.16099955572055]])
# volume: 1267.16 Atoms(symbols='Co24Ce8Cu24', pbc=True, cell=[[0.0, 12.179709674299831, 0.0], [6.437624942941068, 0.0, 0.0], [0.0, -5.892843878783837, -16.16099955572055]])
# volume: 1267.16 Atoms(symbols='Ce24Co8Cu24', pbc=True, cell=[[0.0, 12.179709674299831, 0.0], [6.437624942941068, 0.0, 0.0], [0.0, -5.892843878783837, -16.16099955572055]])
