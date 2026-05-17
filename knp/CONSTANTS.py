seed_val = 1709

mycolors = ["#785EF0", "#7DDEB2", "#DC267F", "#FE6100", "#FFB000", "#5F3331"]
translate_filters = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4, "y": 5}
translate_filternos = {v: k for (k, v) in translate_filters.items()}

def get_ADDFLUX_FOR_MAG_CONST(flux_scaler):
    ADDFLUX_FOR_MAG_CONST = 10 * (
        1 - (flux_scaler.data_min_[0])
    )  # arbitrary zero pt to minimize mag nans when converting flux -> mag
    return ADDFLUX_FOR_MAG_CONST
