from duetrna_shared_core.data_utils import parse_complex_feats, read_pkl


def read_processed_pickle(path):
    return read_pkl(path)


def parse_processed_feats(raw):
    return parse_complex_feats(raw)


__all__ = ["read_processed_pickle", "parse_processed_feats"]
