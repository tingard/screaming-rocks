SENSOR_IDS = (90414, 180113, 21115)
RATE = 10E6


def to_voltage(d):
    return d / (2**16) * 10 - 5
