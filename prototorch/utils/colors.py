"""ProtoTorch color utilities"""


def hex_to_rgb(hex_values):
    for v in hex_values:
        v = v.lstrip('#')
        lv = len(v)
        c = [int(v[i:i + lv // 3], 16) for i in range(0, lv, lv // 3)]
        yield c


def rgb_to_hex(rgb_values):
    for v in rgb_values:
        c = "%02x%02x%02x" % tuple(v)
        yield c
