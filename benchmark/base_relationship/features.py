from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .annotators import EdgeAnnotation
from .frames import ResidueFrames, relative_pose_vector, rotmat_to_rotvec


@dataclass(frozen=True)
class RelationRecord:
    pdb_name: str
    i: int
    j: int
    base_i: str
    base_j: str
    relation: str
    family: str
    seq_sep: int
    z_base: np.ndarray | None
    z_sugar: np.ndarray | None
    stack_features: np.ndarray | None


def _base_center(frame: ResidueFrames) -> np.ndarray | None:
    if frame.base_plane is not None:
        return frame.base_plane.trans
    return None


def _base_normal(frame: ResidueFrames) -> np.ndarray | None:
    if frame.base_plane is not None:
        # After FRAME_CONVENTION, the third column is flipped relative to raw normal.
        return frame.base_plane.rot[:, 2]
    return None


def stack_features(a: ResidueFrames, b: ResidueFrames) -> np.ndarray | None:
    center_i = _base_center(a)
    center_j = _base_center(b)
    normal_i = _base_normal(a)
    normal_j = _base_normal(b)
    if center_i is None or center_j is None or normal_i is None or normal_j is None or a.base_plane is None or b.base_plane is None:
        return None
    delta = center_j - center_i
    plane_distance = abs(float(np.dot(delta, normal_i)))
    cos_normal = abs(float(np.dot(normal_i, normal_j)))
    cos_normal = min(1.0, max(-1.0, cos_normal))
    normal_angle = float(np.degrees(np.arccos(cos_normal)))
    inplane = delta - np.dot(delta, normal_i) * normal_i
    inplane_offset = float(np.linalg.norm(inplane))
    rel_rot = a.base_plane.rot.T @ b.base_plane.rot
    rel_vec = rotmat_to_rotvec(rel_rot)
    twist = float(np.degrees(rel_vec[2]))
    center_distance = float(np.linalg.norm(delta))
    return np.array([plane_distance, normal_angle, inplane_offset, twist, center_distance], dtype=np.float64)


def geometry_fallback_edges(frames: list[ResidueFrames]) -> list[EdgeAnnotation]:
    """Generate approximate pair/stack candidates when DSSR/FR3D labels are absent.

    This is not a replacement for DSSR/FR3D in final paper numbers. It is a
    deterministic smoke-test fallback that lets the metric code run on generated
    full-atom PDBs before external annotation is available.
    """
    out: list[EdgeAnnotation] = []
    for i, fi in enumerate(frames):
        for j in range(i + 1, len(frames)):
            fj = frames[j]
            sf = stack_features(fi, fj)
            if sf is None:
                continue
            plane_distance, normal_angle, inplane_offset, _, center_distance = sf.tolist()
            seq_sep = abs(j - i)
            if seq_sep == 1 and plane_distance <= 4.2 and normal_angle <= 35.0 and inplane_offset <= 5.0:
                out.append(EdgeAnnotation(i=i, j=j, relation="stack", source="geometry"))
                continue
            if seq_sep > 3 and center_distance <= 11.0 and normal_angle <= 55.0:
                out.append(EdgeAnnotation(i=i, j=j, relation="pair_candidate", source="geometry"))
    return out


def relation_records_from_edges(
    pdb_name: str,
    frames: list[ResidueFrames],
    edges: list[EdgeAnnotation],
) -> list[RelationRecord]:
    records = []
    for edge in edges:
        if edge.i < 0 or edge.j < 0 or edge.i >= len(frames) or edge.j >= len(frames):
            continue
        fi, fj = frames[edge.i], frames[edge.j]
        z_base = None
        if fi.base_plane is not None and fj.base_plane is not None:
            z_base = relative_pose_vector(fi.base_plane, fj.base_plane)
        z_sugar = None
        if fi.sugar_gs is not None and fj.sugar_gs is not None:
            z_sugar = relative_pose_vector(fi.sugar_gs, fj.sugar_gs)
        sf = stack_features(fi, fj)
        if z_base is None and z_sugar is None and sf is None:
            continue
        records.append(
            RelationRecord(
                pdb_name=pdb_name,
                i=edge.i,
                j=edge.j,
                base_i=fi.residue.resname,
                base_j=fj.residue.resname,
                relation=edge.relation,
                family=edge.family,
                seq_sep=abs(edge.j - edge.i),
                z_base=z_base,
                z_sugar=z_sugar,
                stack_features=sf,
            )
        )
    return records


def records_to_jsonable(records: list[RelationRecord]) -> list[dict]:
    out = []
    for rec in records:
        out.append(
            {
                "pdb_name": rec.pdb_name,
                "i": rec.i,
                "j": rec.j,
                "base_i": rec.base_i,
                "base_j": rec.base_j,
                "relation": rec.relation,
                "family": rec.family,
                "seq_sep": rec.seq_sep,
                "z_base": None if rec.z_base is None else rec.z_base.tolist(),
                "z_sugar": None if rec.z_sugar is None else rec.z_sugar.tolist(),
                "stack_features": None if rec.stack_features is None else rec.stack_features.tolist(),
            }
        )
    return out


def records_from_jsonable(rows: list[dict]) -> list[RelationRecord]:
    records = []
    for row in rows:
        records.append(
            RelationRecord(
                pdb_name=str(row["pdb_name"]),
                i=int(row["i"]),
                j=int(row["j"]),
                base_i=str(row["base_i"]),
                base_j=str(row["base_j"]),
                relation=str(row["relation"]),
                family=str(row["family"]),
                seq_sep=int(row["seq_sep"]),
                z_base=None if row.get("z_base") is None else np.asarray(row["z_base"], dtype=np.float64),
                z_sugar=None if row.get("z_sugar") is None else np.asarray(row["z_sugar"], dtype=np.float64),
                stack_features=None
                if row.get("stack_features") is None
                else np.asarray(row["stack_features"], dtype=np.float64),
            )
        )
    return records
