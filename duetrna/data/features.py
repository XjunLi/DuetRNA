"""
Feature pickle I/O, residue padding, and coordinate centering.

Code adapted from
https://github.com/Profluent-Internships/MMDiff/blob/main/src/data/components/pdb/data_utils.py
"""

from typing import List, Dict, Any
import numpy as np
import collections
import string
import pickle
import os
import io
import torch.nn.functional as F
import torch

from duetrna.chemistry import nucleotide_constants, protein_constants

# Global map from chain characters to integers.
ALPHANUMERIC = string.ascii_letters + string.digits + ' '
CHAIN_TO_INT = {
    chain_char: i for i, chain_char in enumerate(ALPHANUMERIC)
}


UNPADDED_FEATS = [
    "t",
    "rot_score_scaling",
    "trans_score_scaling",
    "t_seq",
    "t_struct",
]
RIGID_FEATS = ["rigids_0", "rigids_t"]
PAIR_FEATS = ["rel_rots"]


def pad(x: np.ndarray, max_len: int, pad_idx=0, use_torch=False, reverse=False):
    """Right pads dimension of numpy array.

    Args:
        x: numpy like array to pad.
        max_len: desired length after padding
        pad_idx: dimension to pad.
        use_torch: use torch padding method instead of numpy.

    Returns:
        x with its pad_idx dimension padded to max_len
    """
    # Pad only the residue dimension.
    seq_len = x.shape[pad_idx]
    pad_amt = max_len - seq_len
    pad_widths = [(0, 0)] * x.ndim
    if pad_amt < 0:
        raise ValueError(f"Invalid pad amount {pad_amt}")
    if reverse:
        pad_widths[pad_idx] = (pad_amt, 0)
    else:
        pad_widths[pad_idx] = (0, pad_amt)
    if use_torch:
        flat_pad = []
        for before, after in reversed(pad_widths):
            flat_pad.extend([before, after])
        return F.pad(x, tuple(flat_pad))
    return np.pad(x, pad_widths)


def pad_rigid(rigid, max_len):
    from duetrna.geometry.rigid import Rigid

    tensor_7 = rigid.to_tensor_7()
    padded = pad(tensor_7, max_len, use_torch=True)
    return Rigid.from_tensor_7(padded)

def pad_feats(raw_feats, max_len, use_torch=False):
    padded_feats = {
        feat_name: pad(feat, max_len, use_torch=use_torch)
        for feat_name, feat in raw_feats.items()
        if feat_name not in UNPADDED_FEATS + RIGID_FEATS
    }
    for feat_name in PAIR_FEATS:
        if feat_name in padded_feats:
            padded_feats[feat_name] = pad(padded_feats[feat_name], max_len, pad_idx=1)
    for feat_name in UNPADDED_FEATS:
        if feat_name in raw_feats:
            padded_feats[feat_name] = raw_feats[feat_name]
    for feat_name in RIGID_FEATS:
        if feat_name in raw_feats:
            padded_feats[feat_name] = pad_rigid(raw_feats[feat_name], max_len)
    return padded_feats

class CPU_Unpickler(pickle.Unpickler):
    """Pytorch pickle loading workaround.

    https://github.com/pytorch/pytorch/issues/16797
    """
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


def write_pkl(
        save_path: str, pkl_data: Any, create_dir: bool = False, use_torch=False):
    """Serialize data into a pickle file."""
    if create_dir:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if use_torch:
        torch.save(pkl_data, save_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(save_path, 'wb') as handle:
            pickle.dump(pkl_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pkl(read_path: str, verbose=True, use_torch=False, map_location=None):
    """Read data from a pickle file."""
    try:
        if use_torch:
            return torch.load(read_path, map_location=map_location)
        else:
            with open(read_path, 'rb') as handle:
                return pickle.load(handle)
    except Exception as e:
        try:
            with open(read_path, 'rb') as handle:
                return CPU_Unpickler(handle).load()
        except Exception as e2:
            if verbose:
                print(f'Failed to read {read_path}. First error: {e}\n Second error: {e2}')
            raise(e)

def int_id_to_str_id(num: int) -> str:
    """Encodes a number as a string, using reverse spreadsheet style naming.

    Args:
      num: A positive integer.

    Returns:
      A string that encodes the positive integer using reverse spreadsheet style,
      naming e.g. 1 = A, 2 = B, ..., 27 = AA, 28 = BA, 29 = CA, ... This is the
      usual way to encode chain IDs in mmCIF files.
    """
    if num <= 0:
        raise ValueError(f"Only positive integers allowed, got {num}.")

    num = num - 1  # 1-based indexing.
    output = []
    while num >= 0:
        output.append(chr(num % 26 + ord("A")))
        num = num // 26 - 1
    return "".join(output)


def chain_str_to_int(chain_str: str):
    chain_int = 0
    if len(chain_str) == 1:
        return CHAIN_TO_INT[chain_str]
    for i, chain_char in enumerate(chain_str):
        chain_int += CHAIN_TO_INT[chain_char] + (i * len(ALPHANUMERIC))
    return chain_int


def parse_chain_feats_pdb(chain_feats, molecule_constants, molecule_backbone_atom_name, scale_factor=1.0):
    core_atom_idx = molecule_constants.atom_order[molecule_backbone_atom_name]
    chain_feats["bb_mask"] = chain_feats["atom_mask"][:, core_atom_idx]
    bb_pos = chain_feats["atom_positions"][:, core_atom_idx]
    bb_center = np.sum(bb_pos, axis=0) / (np.sum(chain_feats["bb_mask"]) + 1e-5)
    centered_pos = chain_feats["atom_positions"] - bb_center[None, None, :]
    scaled_pos = centered_pos / scale_factor
    chain_feats["atom_positions"] = scaled_pos * chain_feats["atom_mask"][..., None]
    chain_feats["bb_positions"] = chain_feats["atom_positions"][:, core_atom_idx]
    return chain_feats

def get_complex_is_ca_mask(complex_feats):
    is_protein_residue_mask = complex_feats["molecule_type_encoding"][:, 0] == 1
    is_na_residue_mask = (complex_feats["molecule_type_encoding"][:, 1] == 1) | (
        complex_feats["molecule_type_encoding"][:, 2] == 1
    )
    complex_is_ca_mask = np.zeros_like(complex_feats["atom_mask"], dtype=np.bool_)
    complex_is_ca_mask[is_protein_residue_mask, protein_constants.atom_order["CA"]] = True
    complex_is_ca_mask[is_na_residue_mask, nucleotide_constants.atom_order["C4'"]] = True
    return complex_is_ca_mask

def parse_complex_feats(complex_feats, scale_factor=1.0):
    complex_is_ca_mask = get_complex_is_ca_mask(complex_feats)
    complex_feats["bb_mask"] = complex_feats["atom_mask"][complex_is_ca_mask]
    bb_pos = complex_feats["atom_positions"][complex_is_ca_mask]
    bb_center = np.sum(bb_pos, axis=0) / (np.sum(complex_feats["bb_mask"]) + 1e-5)
    centered_pos = complex_feats["atom_positions"] - bb_center[None, None, :]
    scaled_pos = centered_pos / scale_factor
    complex_feats["atom_positions"] = scaled_pos * complex_feats["atom_mask"][..., None]
    complex_feats["bb_positions"] = complex_feats["atom_positions"][complex_is_ca_mask]
    return complex_feats

def concat_np_features(
        np_dicts: List[Dict[str, np.ndarray]], add_batch_dim: bool):
    """Performs a nested concatenation of feature dicts.

    Args:
        np_dicts: list of dicts with the same structure.
            Each dict must have the same keys and numpy arrays as the values.
        add_batch_dim: whether to add a batch dimension to each feature.

    Returns:
        A single dict with all the features concatenated.
    """
    combined_dict = collections.defaultdict(list)
    for chain_dict in np_dicts:
        for feat_name, feat_val in chain_dict.items():
            if add_batch_dim:
                feat_val = feat_val[None]
            combined_dict[feat_name].append(feat_val)
    # Concatenate each feature
    for feat_name, feat_vals in combined_dict.items():
        combined_dict[feat_name] = np.concatenate(feat_vals, axis=0)
    return combined_dict
