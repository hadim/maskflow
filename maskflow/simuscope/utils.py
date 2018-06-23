def get_memory_array(a):
    mem = a.size * a.dtype.itemsize
    return humanbytes(mem)


def humanbytes(b):
    """Return the given bytes as a human friendly KB, mb, gb, or tb string."""
    b = float(b)
    kb = float(1024)
    mb = float(kb**2)  # 1,048,576
    gb = float(kb**3)  # 1,073,741,824
    tb = float(kb**4)  # 1,099,511,627,776

    if b < kb:
        return '{0} {1}'.format(b, 'Bytes' if 0 == b > 1 else 'Byte')
    elif kb <= b < mb:
        return '{0:.2f} KB'.format(b / kb)
    elif mb <= b < gb:
        return '{0:.2f} MB'.format(b / mb)
    elif gb <= b < tb:
        return '{0:.2f} GB'.format(b / gb)
    elif tb <= b:
        return '{0:.2f} TB'.format(b / tb)
