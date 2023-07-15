import torch
from unimatch.geometry import coords_grid
import kornia.geometry.epipolar as epipolar

def pixelwise_e_estimation(grid_coords, matched_coords, camera_matrix):
    """
    estimate the essential matrix pixelwise from the correspondences.
    grid_coords: [B,2,H,W]
    matched_coords: [B,2,H,W]
    """
    b,_,h,w = grid_coords.size()

    # reshape grid and matched coords in kornia shape.
    grid_coords = grid_coords.permute(0,2,3,1).view(b,h*w,2)
    matched_coords = matched_coords.permute(0,2,3,1).view(b,h*w,2)

    fundamental_matrix = epipolar.find_fundamental(grid_coords, matched_coords)
    essential_matrix = epipolar.essential_from_fundamental(fundamental_matrix, camera_matrix, camera_matrix)
    rot, trans, _ = epipolar.motion_from_essential_choose_solution(essential_matrix, camera_matrix, camera_matrix,
                                                                   grid_coords, matched_coords)

    rot = matrix_to_euler_angles(rot, "XYZ")

    return rot, trans

def vo_loss_func(flow_preds, rot_gt, trans_gt, batch_size, device, camera_matrix, tau=0.5, gamma=0.9):
    """
    corr: HxWxHxW (pixelwise correlation b/w img1 and img2)
    """

    n_predictions = len(flow_preds)
    vo_loss = 0.0
    h, w = flow_preds[0].size()[-2:]

    grid_coords = coords_grid(batch_size, h, w).to(device) # [B,2,H,W]

    for i in range(n_predictions):
        # create e_pred from flow_preds.
        flow_pred = flow_preds[i] # [B,2,H,W]
        matched_coords = flow_pred + grid_coords

        # using the matched correspondences, estimate the essential matrix
        # for all pixels.
        rot_estimated, trans_estimated = pixelwise_e_estimation(grid_coords, matched_coords, camera_matrix)
        i_weight = gamma ** (n_predictions - i - 1)

        i_loss = (rot_estimated - rot_gt).norm(dim=1) + tau * (trans_estimated - trans_gt).norm(dim=1)
        vo_loss += i_weight * i_loss.mean()

    return vo_loss

def get_skew_mat(a):
    """
    return skew symmetric matrix of a vector a (3,1).
    """
    return torch.Tensor([[0, -a[2], a[1]],
                         [a[2], 0, -a[0]],
                         [-a[1], a[0], 0]])

def matrix_to_euler_angles(matrix: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)

def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
) -> torch.Tensor:
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])

def _index_from_letter(letter: str) -> int:
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter must be either X, Y or Z.")
