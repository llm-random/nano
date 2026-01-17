from typing import Optional
import torch

from src.core.model import get_init_weight


# 1) Using the SVD (orthogonal projectors)
def svd_op(a):
    A = a
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    r = (S > 1e-12).sum().item()
    Ur = U[:, :r]
    Vr = Vh[:r, :].T

    P1 = Ur @ Ur.T          # m x m
    P2 = Vr @ Vr.T          # n x n
    return P1, P2


# 2) Using the Moore–Penrose pseudoinverse (canonical projectors)
def mpp(a):
    A = a
    Ap = torch.linalg.pinv(A)

    P1 = A @ Ap   # m x m
    P2 = Ap @ A   # n x n
    return P1, P2

def svd_g(a):
    A = a
    assert torch.isfinite(A).all(), "Matrix contains NaN or Inf"
    U, S, Vh = torch.linalg.svd(A, full_matrices=True)
    r = (S > 1e-12).sum().item()
    U_perp = U[:, r:]
    V_perp = Vh[r:, :].T
    L = get_init_weight((A.shape[0], U_perp.shape[1]), fan_in=A.shape[0], init_type="truncated_normal_fixed", scale=1.0).to(A.device)
    K = get_init_weight((V_perp.shape[1], A.shape[1]), fan_in=A.shape[1], init_type="truncated_normal_fixed", scale=1.0).to(A.device)

    P1 = torch.eye(A.shape[0], device=A.device) + L @ U_perp.T
    P2 = torch.eye(A.shape[1], device=A.device) + V_perp @ K
    return P1, P2


def norm_index(idx, device: torch.device):
    # None → full range
    if idx is None:
        return None
    # Slices are device-agnostic
    if isinstance(idx, slice):
        return idx
    # Python lists/tuples/NumPy arrays → LongTensor on device
    if isinstance(idx, (list, tuple)):
        return torch.as_tensor(idx, dtype=torch.long, device=device)
    if torch.is_tensor(idx):
        if idx.dtype == torch.bool:
            # boolean masks must already be correct shape; just move device
            return idx.to(device)
        # integer/long indices
        return idx.to(device=device, dtype=torch.long)
    raise TypeError(f"Unsupported index type: {type(idx)}")

def smart_projections(t, iy, ix, fun=svd_g):
    iy = norm_index(iy, t.device)
    ix = norm_index(ix, t.device)
    assert not (iy is None and ix is None)
    
    if iy is None:
        ar = t[:, ix]
        _, p2ll = fun(t)
        _ = None
        diff_weights = t[:, ix] - t@p2ll[:, ix]
        err = torch.norm(diff_weights)
        if err > 0.01: #dev
            print(f"err: {err}")
            print(t.shape)
        return None, p2ll[:, ix], diff_weights
    elif ix is None:
        p1r, _ = fun(t)
        _, p2ll = fun(p1r[iy]@t)
        _ = None
        diff_weights = t[iy] - p1r[iy]@t
        err = torch.norm(diff_weights)
        if err > 0.01: #dev
            print(f"err: {err}")
            print(t.shape)
        return p1r[iy], None, diff_weights
    else:
        ar = t[:, ix]
        p1r, _ = fun(ar)
        _, p2ll = fun(p1r[iy]@t)
        _ = None
        diff_weights = t[iy][:, ix] - p1r[iy]@t@p2ll[:, ix]
        err = torch.norm(diff_weights)
        print(err)
        if err > 0.01: #dev
            print(f"err: {err}")
            print(t.shape)
        return p1r[iy], p2ll[:, ix], diff_weights


def transfer_selected(
    W_from: torch.Tensor,
    W: torch.Tensor,
    iy: Optional[torch.Tensor] = None,
    ix: Optional[torch.Tensor] = None,
):
    """
    Copy selected rows/columns/submatrix from A → B.

    - If both proj_in_topk_indices and proj_out_topk_indices are given:
        copy submatrix A[R, C] → B[R, C]  (Cartesian product of indices)
    - If only proj_in_topk_indices: copy rows
    - If only proj_out_topk_indices: copy columns
    """
    device = W.device
    iy = norm_index(iy, device)
    ix = norm_index(ix, device)
    W_from = W_from.to(device)

    # Both rows and columns: submatrix
    if iy is not None and ix is not None:
        # Create index grids for Cartesian product R × C
        rows, cols = torch.meshgrid(
            iy, ix, indexing="ij"
        )
        W[rows, cols] = W_from[rows, cols]
        return W

    # Only rows
    if iy is not None:
        W[iy, :] = W_from[iy, :]
        return W

    # Only columns
    if ix is not None:
        W[:, ix] = W_from[:, ix]
        return W

    return W