from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch


RNA_CHAR_TO_AATYPE9 = {
    "A": 4,
    "C": 5,
    "G": 6,
    "U": 7,
    "T": 7,
}


def parse_fasta_records(path: str) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    current_name = None
    current_seq: List[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_name is not None:
                    records.append((current_name, "".join(current_seq)))
                current_name = line[1:].strip() or f"seq_{len(records)}"
                current_seq = []
            else:
                current_seq.append(line.replace(" ", "").upper())
    if current_name is not None:
        records.append((current_name, "".join(current_seq)))
    return records


def encode_rna_sequence(sequence: str) -> torch.Tensor:
    seq = sequence.strip().upper().replace(" ", "")
    if not seq:
        raise ValueError("Empty RNA sequence is not allowed.")
    encoded = []
    for idx, ch in enumerate(seq):
        if ch not in RNA_CHAR_TO_AATYPE9:
            raise ValueError(
                f"Unsupported nucleotide `{ch}` at position {idx + 1} in sequence `{sequence}`. "
                "Only A/C/G/U/T are supported."
            )
        encoded.append(RNA_CHAR_TO_AATYPE9[ch])
    return torch.tensor(encoded, dtype=torch.long)


def load_sequence_records(fasta_path: str | None, inline_sequence: str | None, inline_name: str) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    if fasta_path:
        fasta = Path(fasta_path)
        if not fasta.exists():
            raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
        records.extend(parse_fasta_records(str(fasta)))
    if inline_sequence:
        records.append((inline_name, str(inline_sequence)))
    if not records:
        raise ValueError("Expected `fasta_path` or `inline_sequence`.")
    return records


__all__ = ["RNA_CHAR_TO_AATYPE9", "encode_rna_sequence", "load_sequence_records", "parse_fasta_records"]
