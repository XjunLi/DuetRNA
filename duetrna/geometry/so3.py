from typing import Tuple

import numpy as np
import torch



def scale_rotmat(
    rotation_matrix: torch.Tensor, scalar: torch.Tensor, tol: float = 1e-7
) -> torch.Tensor:
    """
    Scale rotation matrix. This is done by converting it to vector representation,
    scaling the length of the vector and converting back to matrix representation.

    Args:
        rotation_matrix: Rotation matrices.
        scalar: Scalar values used for scaling. Should have one fewer dimension than the
            rotation matrices for correct broadcasting.
        tol: Numerical offset for stability.

    Returns:
        Scaled rotation matrix.
    """
    # Check whether dimensions match.
    assert rotation_matrix.ndim - 1 == scalar.ndim
    scaled_rmat = rotvec_to_rotmat(rotmat_to_rotvec(rotation_matrix) * scalar, tol=tol)
    return scaled_rmat


def _broadcast_identity(target: torch.Tensor) -> torch.Tensor:
    """
    Generate a 3 by 3 identity matrix and broadcast it to a batch of target matrices.

    Args:
        target (torch.Tensor): Batch of target 3 by 3 matrices.

    Returns:
        torch.Tensor: 3 by 3 identity matrices in the shapes of the target.
    """
    id3 = torch.eye(3, device=target.device, dtype=target.dtype)
    id3 = torch.broadcast_to(id3, target.shape)
    return id3


def skew_matrix_exponential_map_axis_angle(
    angles: torch.Tensor, skew_matrices: torch.Tensor
) -> torch.Tensor:
    """
    Compute the matrix exponential of a rotation in axis-angle representation with the axis in skew
    matrix representation form. Maps the rotation from the lie group to the rotation matrix
    representation. Uses Rodrigues' formula instead of `torch.linalg.matrix_exp` for better
    computational performance:

    .. math::

        \exp(\theta \mathbf{K}) = \mathbf{I} + \sin(\theta) \mathbf{K} + [1 - \cos(\theta)] \mathbf{K}^2

    Args:
        angles (torch.Tensor): Batch of rotation angles.
        skew_matrices (torch.Tensor): Batch of rotation axes in skew matrix (lie so(3)) basis.

    Returns:
        torch.Tensor: Batch of corresponding rotation matrices.
    """
    # Set up identity matrix and broadcast.
    id3 = _broadcast_identity(skew_matrices)

    # Broadcast angle vector to right dimensions
    angles = angles[..., None, None]

    exp_skew = (
        id3
        + torch.sin(angles) * skew_matrices
        + (1.0 - torch.cos(angles))
        * torch.einsum("b...ik,b...kj->b...ij", skew_matrices, skew_matrices)
    )
    return exp_skew


def skew_matrix_exponential_map(
    angles: torch.Tensor, skew_matrices: torch.Tensor, tol=1e-7
) -> torch.Tensor:
    """
    Compute the matrix exponential of a rotation vector in skew matrix representation. Maps the
    rotation from the lie group to the rotation matrix representation. Uses the following form of
    Rodrigues' formula instead of `torch.linalg.matrix_exp` for better computational performance
    (in this case the skew matrix already contains the angle factor):

    .. math ::

        \exp(\mathbf{K}) = \mathbf{I} + \frac{\sin(\theta)}{\theta} \mathbf{K} + \frac{1-\cos(\theta)}{\theta^2} \mathbf{K}^2

    This form has the advantage, that Taylor expansions can be used for small angles (instead of
    having to compute the unit length axis by dividing the rotation vector by small angles):

    .. math ::

        \frac{\sin(\theta)}{\theta} \approx 1 - \frac{\theta^2}{6}
        \frac{1-\cos(\theta)}{\theta^2} \approx \frac{1}{2} - \frac{\theta^2}{24}

    Args:
        angles (torch.Tensor): Batch of rotation angles.
        skew_matrices (torch.Tensor): Batch of rotation axes in skew matrix (lie so(3)) basis.

    Returns:
        torch.Tensor: Batch of corresponding rotation matrices.
    """
    # Set up identity matrix and broadcast.
    id3 = _broadcast_identity(skew_matrices)

    # Broadcast angles and pre-compute square.
    angles = angles[..., None, None]
    angles_sq = angles.square()

    mask_zero = torch.abs(angles) < tol
    safe_angles = torch.where(mask_zero, torch.ones_like(angles), angles)
    safe_angles_sq = torch.where(mask_zero, torch.ones_like(angles_sq), angles_sq)

    # Avoid forming 0/0 in masked branches. `torch.where` still keeps both branches
    # in the autograd graph, so unsafe unused branches can poison backward.
    sin_coeff = torch.sin(angles) / safe_angles
    cos_coeff = (1.0 - torch.cos(angles)) / safe_angles_sq
    # Use second order Taylor expansion for values close to zero.
    sin_coeff_small = 1.0 - angles_sq / 6.0
    cos_coeff_small = 0.5 - angles_sq / 24.0

    sin_coeff = torch.where(mask_zero, sin_coeff_small, sin_coeff)
    cos_coeff = torch.where(mask_zero, cos_coeff_small, cos_coeff)

    # Compute matrix exponential using Rodrigues' formula.
    exp_skew = (
        id3
        + sin_coeff * skew_matrices
        + cos_coeff * torch.einsum("b...ik,b...kj->b...ij", skew_matrices, skew_matrices)
    )
    return exp_skew


def rotvec_to_rotmat(rotation_vectors: torch.Tensor, tol: float = 1e-7) -> torch.Tensor:
    """
    Convert rotation vectors to rotation matrix representation. The length of the rotation vector
    is the angle of rotation, the unit vector the rotation axis.

    Args:
        rotation_vectors (torch.Tensor): Batch of rotation vectors.
        tol: small offset for numerical stability.

    Returns:
        torch.Tensor: Rotation in rotation matrix representation.
    """
    # Compute rotation angle as vector norm.
    rotation_angles = torch.norm(rotation_vectors, dim=-1)

    # Map axis to skew matrix basis.
    skew_matrices = vector_to_skew_matrix(rotation_vectors)

    # Compute rotation matrices via matrix exponential.
    rotation_matrices = skew_matrix_exponential_map(rotation_angles, skew_matrices, tol=tol)

    return rotation_matrices


def rotmat_to_rotvec(rotation_matrices: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of rotation matrices to rotation vectors (logarithmic map from SO(3) to so(3)).
    The standard logarithmic map can be derived from Rodrigues' formula via Taylor approximation
    (in this case operating on the vector coefficients of the skew so(3) basis).

    ..math ::

        \left[\log(\mathbf{R})\right]^\lor = \frac{\theta}{2\sin(\theta)} \left[\mathbf{R} - \mathbf{R}^\top\right]^\lor

    This formula has problems at 1) angles theta close or equal to zero and 2) at angles close and
    equal to pi.

    To improve numerical stability for case 1), the angle term at small or zero angles is
    approximated by its truncated Taylor expansion:

    .. math ::

        \left[\log(\mathbf{R})\right]^\lor \approx \frac{1}{2} (1 + \frac{\theta^2}{6}) \left[\mathbf{R} - \mathbf{R}^\top\right]^\lor

    For angles close or equal to pi (case 2), the outer product relation can be used to obtain the
    squared rotation vector:

    .. math :: \omega \otimes \omega = \frac{1}{2}(\mathbf{I} + R)

    Taking the root of the diagonal elements recovers the normalized rotation vector up to the signs
    of the component. The latter can be obtained from the off-diagonal elements.

    Adapted from https://github.com/jasonkyuyim/se3_diffusion/blob/2cba9e09fdc58112126a0441493b42022c62bbea/data/so3_utils.py
    which was adapted from https://github.com/geomstats/geomstats/blob/master/geomstats/geometry/special_orthogonal.py
    with heavy help from https://cvg.cit.tum.de/_media/members/demmeln/nurlanov2021so3log.pdf

    Args:
        rotation_matrices (torch.Tensor): Input batch of rotation matrices.

    Returns:
        torch.Tensor: Batch of rotation vectors.
    """
    # Get angles and sin/cos from rotation matrix.
    angles, angles_sin, _ = angle_from_rotmat(rotation_matrices)
    # Compute skew matrix representation and extract so(3) vector components.
    vector = skew_matrix_to_vector(rotation_matrices - rotation_matrices.transpose(-2, -1))

    # Three main cases for angle theta.
    # 1) Angle is close to zero -> use Taylor series.
    mask_zero = torch.isclose(angles, torch.zeros_like(angles))
    # 2) Angle is close to pi -> use outer product relation.
    mask_pi = torch.isclose(angles, torch.full_like(angles, np.pi), atol=1e-2)

    # Compute case dependent pre-factor without constructing singular masked branches.
    prefactor_small = 0.5 * (1.0 + angles**2 / 6.0)
    safe_angles_sin = torch.where(
        torch.abs(angles_sin) < 1e-7,
        torch.ones_like(angles_sin),
        angles_sin,
    )
    prefactor_standard = angles / (2.0 * safe_angles_sin)
    prefactor = torch.where(mask_zero, prefactor_small, prefactor_standard)
    prefactor = torch.where(mask_pi, torch.zeros_like(prefactor), prefactor)
    vector = vector * prefactor[..., None]

    # For angles close to pi, derive vectors from their outer product (ww' = 1 + R).
    id3 = _broadcast_identity(rotation_matrices)
    skew_outer = (id3 + rotation_matrices) / 2.0
    # Ensure diagonal is >= 0 for square root (uses identity for masking).
    skew_outer = skew_outer + (torch.relu(skew_outer) - skew_outer) * id3

    # Get basic rotation vector as sqrt of diagonal (is unit vector). Clamp with
    # epsilon so exact zero diagonal entries do not create infinite sqrt gradients.
    vector_pi = torch.sqrt(torch.clamp(torch.diagonal(skew_outer, dim1=-2, dim2=-1), min=1e-12))

    # Compute the signs of vector elements (up to a global phase).
    # Fist select indices for outer product slices with the largest norm.
    signs_line_idx = torch.argmax(torch.norm(skew_outer, dim=-1), dim=-1).long()
    # Select rows of outer product and determine signs.
    signs_line = torch.take_along_dim(skew_outer, dim=-2, indices=signs_line_idx[..., None, None])
    signs_line = signs_line.squeeze(-2)
    signs = torch.sign(signs_line)

    # Apply signs and rotation vector.
    vector_pi = vector_pi * angles[..., None] * signs

    # Fill entries for angle == pi in rotation vector (basic vector has zero entries at this point).
    vector = vector + vector_pi * mask_pi.to(vector.dtype)[..., None]

    return vector


def angle_from_rotmat(
    rotation_matrices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute rotation angles (as well as their sines and cosines) encoded by rotation matrices.
    Uses atan2 for better numerical stability for small angles.

    Args:
        rotation_matrices (torch.Tensor): Batch of rotation matrices.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Batch of computed angles, sines of the
          angles and cosines of angles.
    """
    # Compute sine of angles (uses the relation that the unnormalized skew vector generated by a
    # rotation matrix has the length 2*sin(theta))
    skew_matrices = rotation_matrices - rotation_matrices.transpose(-2, -1)
    skew_vectors = skew_matrix_to_vector(skew_matrices)
    angles_sin = torch.norm(skew_vectors, dim=-1) / 2.0
    # Compute the cosine of the angle using the relation cos theta = 1/2 * (Tr[R] - 1)
    angles_cos = (torch.einsum("...ii", rotation_matrices) - 1.0) / 2.0

    # Compute angles using the more stable atan2
    angles = torch.atan2(angles_sin, angles_cos)

    return angles, angles_sin, angles_cos


def vector_to_skew_matrix(vectors: torch.Tensor) -> torch.Tensor:
    """
    Map a vector into the corresponding skew matrix so(3) basis.
    ```
                [  0 -z  y]
    [x,y,z] ->  [  z  0 -x]
                [ -y  x  0]
    ```

    Args:
        vectors (torch.Tensor): Batch of vectors to be mapped to skew matrices.

    Returns:
        torch.Tensor: Vectors in skew matrix representation.
    """
    # Generate empty skew matrices.
    skew_matrices = torch.zeros((*vectors.shape, 3), device=vectors.device, dtype=vectors.dtype)

    # Populate positive values.
    skew_matrices[..., 2, 1] = vectors[..., 0]
    skew_matrices[..., 0, 2] = vectors[..., 1]
    skew_matrices[..., 1, 0] = vectors[..., 2]

    # Generate skew symmetry.
    skew_matrices = skew_matrices - skew_matrices.transpose(-2, -1)

    return skew_matrices


def skew_matrix_to_vector(skew_matrices: torch.Tensor) -> torch.Tensor:
    """
    Extract a rotation vector from the so(3) skew matrix basis.

    Args:
        skew_matrices (torch.Tensor): Skew matrices.

    Returns:
        torch.Tensor: Rotation vectors corresponding to skew matrices.
    """
    vectors = torch.zeros_like(skew_matrices[..., 0])
    vectors[..., 0] = skew_matrices[..., 2, 1]
    vectors[..., 1] = skew_matrices[..., 0, 2]
    vectors[..., 2] = skew_matrices[..., 1, 0]
    return vectors


def _rotquat_to_axis_angle(
    rotation_quaternions: torch.Tensor, tol: float = 1e-7
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Auxiliary routine for computing rotation angle and rotation axis from unit quaternions. To avoid
    complications, rotations vectors with angles below `tol` are set to zero.

    Args:
        rotation_quaternions (torch.Tensor): Rotation quaternions in [r, i, j, k] format.
        tol (float, optional): Threshold for small rotations. Defaults to 1e-7.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Rotation angles and axes.
    """
    # Compute rotation axis and normalize (accounting for small length axes).
    rotation_axes = rotation_quaternions[..., 1:]
    rotation_axes_norms = torch.norm(rotation_axes, dim=-1)

    # Compute rotation angle via atan2
    rotation_angles = 2.0 * torch.atan2(rotation_axes_norms, rotation_quaternions[..., 0])

    # Save division.
    rotation_axes = rotation_axes / (rotation_axes_norms[:, None] + tol)
    return rotation_angles, rotation_axes


def rotquat_to_rotvec(rotation_quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert unit quaternions to rotation vectors.

    Args:
        rotation_quaternions (torch.Tensor): Input quaternions in [r,i,j,k] format.

    Returns:
        torch.Tensor: Rotation vectors.
    """
    rotation_angles, rotation_axes = _rotquat_to_axis_angle(rotation_quaternions)
    rotation_vectors = rotation_axes * rotation_angles[..., None]
    return rotation_vectors


def rotquat_to_rotmat(rotation_quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert unit quaternion to rotation matrix.

    Args:
        rotation_quaternions (torch.Tensor): Input quaternions in [r,i,j,k] format.

    Returns:
        torch.Tensor: Rotation matrices.
    """
    rotation_angles, rotation_axes = _rotquat_to_axis_angle(rotation_quaternions)
    skew_matrices = vector_to_skew_matrix(rotation_axes * rotation_angles[..., None])
    rotation_matrices = skew_matrix_exponential_map(rotation_angles, skew_matrices)
    return rotation_matrices


def apply_rotvec_to_rotmat(
    rotation_matrices: torch.Tensor,
    rotation_vectors: torch.Tensor,
    tol: float = 1e-7,
) -> torch.Tensor:
    """
    Update a rotation encoded in a rotation matrix with a rotation vector.

    Args:
        rotation_matrices: Input batch of rotation matrices.
        rotation_vectors: Input batch of rotation vectors.
        tol: Small offset for numerical stability.

    Returns:
        Updated rotation matrices.
    """
    # Convert vector to matrices.
    rmat_right = rotvec_to_rotmat(rotation_vectors, tol=tol)
    # Accumulate rotation.
    rmat_rotated = torch.einsum("...ij,...jk->...ik", rotation_matrices, rmat_right)
    return rmat_rotated


def rotmat_to_skew_matrix(mat: torch.Tensor) -> torch.Tensor:
    """
    Generates skew matrix for corresponding rotation matrix.

    Args:
        mat (torch.Tensor): Batch of rotation matrices.

    Returns:
        torch.Tensor: Skew matrices in the shapes of mat.
    """
    vec = rotmat_to_rotvec(mat)
    return vector_to_skew_matrix(vec)


def skew_matrix_to_rotmat(skew: torch.Tensor) -> torch.Tensor:
    """
    Generates rotation matrix for corresponding skew matrix.

    Args:
        skew (torch.Tensor): Batch of target 3 by 3 skew symmetric matrices.

    Returns:
        torch.Tensor: Rotation matrices in the shapes of skew.
    """
    vec = skew_matrix_to_vector(skew)
    return rotvec_to_rotmat(vec)


def local_log(point: torch.Tensor, base_point: torch.Tensor) -> torch.Tensor:
    """
    Matrix logarithm. Computes left-invariant vector field of beinging base_point to point
    on the manifold. Follows the signature of geomstats' equivalent function.
    https://geomstats.github.io/api/geometry.html#geomstats.geometry.lie_group.MatrixLieGroup.log

    Args:
        point (torch.Tensor): Batch of rotation matrices to compute vector field at.
        base_point (torch.Tensor): Transport coordinates to take matrix logarithm.

    Returns:
        torch.Tensor: Skew matrix that holds the vector field (in the tangent space).
    """
    return rotmat_to_skew_matrix(rot_mult(rot_transpose(base_point), point))


def multidim_trace(mat: torch.Tensor) -> torch.Tensor:
    """Take the trace of a matrix with leading dimensions."""
    return torch.einsum("...ii->...", mat)


def geodesic_dist(mat_1: torch.Tensor, mat_2: torch.Tensor) -> torch.Tensor:
    """
    Calculate the geodesic distance of two rotation matrices.

    Args:
        mat_1 (torch.Tensor): First rotation matrix.
        mat_2 (torch.Tensor): Second rotation matrix.

    Returns:
        Scalar for the geodesic distance between mat_1 and mat_2 with the same
        leading (i.e. batch) dimensions.
    """
    A = rotmat_to_skew_matrix(rot_mult(rot_transpose(mat_1), mat_2))
    return torch.sqrt(multidim_trace(rot_mult(A, rot_transpose(A))))


def rot_transpose(mat: torch.Tensor) -> torch.Tensor:
    """Take the transpose of the last two dimensions."""
    return torch.transpose(mat, -1, -2)


def rot_mult(mat_1: torch.Tensor, mat_2: torch.Tensor) -> torch.Tensor:
    """Matrix multiply two rotation matrices with leading dimensions."""
    return torch.einsum("...ij,...jk->...ik", mat_1, mat_2)


def calc_rot_vf(mat_t: torch.Tensor, mat_1: torch.Tensor) -> torch.Tensor:
    """
    Computes the vector field Log_{mat_t}(mat_1).

    Args:
        mat_t (torch.Tensor): base point to compute vector field at.
        mat_1 (torch.Tensor): target rotation.

    Returns:
        Rotation vector representing the vector field.
    """
    return rotmat_to_rotvec(rot_mult(rot_transpose(mat_t), mat_1))


def geodesic_t(t: float, mat: torch.Tensor, base_mat: torch.Tensor, rot_vf=None) -> torch.Tensor:
    """
    Computes the geodesic at time t. Specifically, R_t = Exp_{base_mat}(t * Log_{base_mat}(mat)).

    Args:
        t: time along geodesic.
        mat: target points on manifold.
        base_mat: source point on manifold.

    Returns:
        Point along geodesic starting at base_mat and ending at mat.
    """
    if rot_vf is None:
        rot_vf = calc_rot_vf(base_mat, mat)
    mat_t = rotvec_to_rotmat(t * rot_vf)
    if base_mat.shape != mat_t.shape:
        raise ValueError(
            f'Incompatible shapes: base_mat={base_mat.shape}, mat_t={mat_t.shape}')
    return torch.einsum("...ij,...jk->...ik", base_mat, mat_t)
