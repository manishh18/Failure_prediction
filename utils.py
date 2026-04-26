def kelvin_to_celsius(k_temp):
    return k_temp - 273.15

def ordinal_encoding(X):
    mapping = {"L": 0, "M": 1, "H": 2}
    return X.replace(mapping)