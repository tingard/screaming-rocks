SENSOR_IDS = (90414, 180113, 21115)

def to_voltage(d):
    return d / (2**16) * 10 - 5
