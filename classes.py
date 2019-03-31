import re


def remove_junk_str(input):
    out = re.sub(' +', '_', re.sub(r'([^\s\w]|_)+', '', input.lower()))
    return out


def get_classes():
    classes = '''Continuous urban fabric (S.L. : > 80%)
Discontinuous dense urban fabric (S.L. : 50% -  80%)
Discontinuous medium density urban fabric (S.L. : 30% - 50%)
Discontinuous low density urban fabric (S.L. : 10% - 30%)
Discontinuous very low density urban fabric (S.L. : < 10%)
Isolated structures
Industrial, commercial, public, military and private units
Port areas
Airports
Mineral extraction and dump sites
Construction sites
Land without current use
Green urban areas
Sports and leisure facilities
Arable land (annual crops)
Permanent crops (vineyards, fruit trees, olive groves)
Pastures
Complex and mixed cultivation patterns
Forests
Herbaceous vegetation associations (natural grassland, moors...)
Open spaces with little or no vegetation (beaches, dunes, bare rocks, glaciers)
Wetlands
Water
Other roads and associated land
Fast transit roads and associated land
Railways and associated land
No data clouds and shadows'''
    cleaned_classes_list = remove_junk_str(classes).split("\n")
    return cleaned_classes_list