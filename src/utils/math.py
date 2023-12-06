def interopolate_color(initial_color: tuple, end_color: tuple, t: float):
    """
    Interopolate between two colors
    initial_color: tuple of three values, each between 0 and 1
    end_color: tuple of three values, each between 0 and 1
    t: a value between 0 and 1
    """
    return (
        t * initial_color[0] + (1-t) * end_color[0],
        t * initial_color[1] + (1-t) * end_color[1],
        t * initial_color[2] + (1-t) * end_color[2]
    )


def interopolate_float(initial_value: float, end_value: float, t: float):
    """
    Interopolate between two floats
    initial_value: float
    end_value: float
    t: a value between 0 and 1
    """
    return t * initial_value + (1-t) * end_value
