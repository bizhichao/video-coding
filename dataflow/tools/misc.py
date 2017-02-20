import glob
import os
import re


def n_positive_integers(n, value):
    n_orig = n
    n = int(n)
    if n < 1 or n != n_orig:
        raise ValueError('n must be a positive integer')

    try:
        value = int(value)
    except (TypeError, ValueError):
        sequence_len = len(value)
        if sequence_len != n:
            raise ValueError(
                'Expected sequence of %d positive integers, but received %r' %
                (n, value))
        try:
            values = tuple(int(x) for x in value)
        except:
            raise ValueError(
                'Expected sequence of %d positive integers, but received %r' %
                (n, value))
        for x in values:
            if x < 1:
                raise ValueError('expected positive integer, but received %d' % x)
        return values

    if value < 1:
        raise ValueError('expected positive integer, but received %d' % value)
    return (value,) * n


def get_filenames_from_dir(data_dir, file_fmts):
    assert os.path.isdir(data_dir)
    if isinstance(file_fmts, str):
        file_fmts = [file_fmts]

    data_path = []
    for file_fmt in file_fmts:
        data_path += glob.glob(os.path.abspath(os.path.join(data_dir, file_fmt)))
    return data_path


def get_width_height_from_name(paths_or_names):
    """paths is transform_matrix path string or list of path strings"""
    if paths_or_names is None:
        return None

    paths_or_names = [paths_or_names] if not isinstance(paths_or_names, (list, tuple)) else paths_or_names

    pattern = re.compile('[0-9]+x[0-9]+')
    shapes = []
    for path in paths_or_names:
        size = pattern.findall(path)
        assert len(size) == 1, 'Shape must be inferred from name'
        shapes.append(list(map(int, size[0].split('x'))))
    return shapes[0] if len(paths_or_names) == 1 else shapes
