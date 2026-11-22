from __future__ import annotations

from copy import deepcopy
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple
import warnings

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from NO.modules.point_encoding import PointCloudPerceiverChannelsEncoder
from NO.modules.point_position_embedding import PosEmbLinear
from NO.modules.transformer import MLP, ResidualCrossAttentionBlock


# -----------------------------------------------------------------------------
# Device
# -----------------------------------------------------------------------------

def _select_torch_device() -> torch.device:
    """
    Select a usable device, not merely a visible CUDA device.

    CUDA 13 wheels can report ``is_available() == True`` on Pascal GPUs even
    though those wheels contain no Pascal kernels.  A tiny synchronized tensor
    operation catches that mismatch before a long training run reaches its
    first batch.  Set NO_DEVICE=cpu/cuda to override auto selection.
    """
    requested = os.environ.get("NO_DEVICE", "auto").strip().lower()
    if requested not in {"auto", "cpu", "cuda", "cuda:0"}:
        raise ValueError(
            f"Invalid NO_DEVICE={requested!r}; expected auto, cpu, cuda, or cuda:0."
        )
    if requested == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        if requested.startswith("cuda"):
            raise RuntimeError("NO_DEVICE requests CUDA, but torch.cuda.is_available() is False.")
        return torch.device("cpu")

    cuda_device = torch.device("cuda:0")
    try:
        probe = torch.zeros(1, device=cuda_device)
        probe.add_(1.0)
        torch.cuda.synchronize(cuda_device)
        del probe
        return cuda_device
    except Exception as exc:
        try:
            gpu_name = torch.cuda.get_device_name(0)
            capability = torch.cuda.get_device_capability(0)
            compiled_arches = torch.cuda.get_arch_list()
        except Exception:
            gpu_name, capability, compiled_arches = "unknown", "unknown", []
        message = (
            "CUDA is visible but unusable by this PyTorch wheel. "
            f"GPU={gpu_name}, capability={capability}, "
            f"torch={torch.__version__}, compiled_arches={compiled_arches}. "
            "Install a PyTorch wheel that contains this GPU architecture "
            "(Pascal GPUs require a CUDA 12.6-or-earlier wheel), or run with "
            "NO_DEVICE=cpu / --device cpu."
        )
        if requested.startswith("cuda"):
            raise RuntimeError(message) from exc
        warnings.warn(message + " Falling back to CPU.", RuntimeWarning, stacklevel=2)
        return torch.device("cpu")


device = _select_torch_device()


# -----------------------------------------------------------------------------
# Dataset and collate
# -----------------------------------------------------------------------------

class PlateSampleDataset(Dataset):
    def __init__(self, samples: Sequence[Dict[str, Any]]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def _as_tensor(x: Any, dtype=torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype)
    return torch.as_tensor(x, dtype=dtype)


def _pad_float_list(
    tensors: List[torch.Tensor],
    *,
    padding_value: float,
) -> torch.Tensor:
    return pad_sequence(tensors, batch_first=True, padding_value=padding_value)


def _infer_query_from_sample(sample: Dict[str, Any]) -> torch.Tensor:
    if "query" in sample:
        return _as_tensor(sample["query"])
    if "pc" in sample:
        return _as_tensor(sample["pc"])[..., :3]
    raise KeyError("Sample must contain either 'query' or 'pc'.")


def _infer_query_net_from_sample(sample: Dict[str, Any]) -> torch.Tensor:
    if "query_net" in sample:
        return _as_tensor(sample["query_net"])
    if "query" in sample:
        return _as_tensor(sample["query"])
    return _as_tensor(sample["pc"])[..., :3]


def _infer_pc_net_from_sample(sample: Dict[str, Any]) -> torch.Tensor:
    if "pc_net" in sample:
        return _as_tensor(sample["pc_net"])
    return _as_tensor(sample["pc"])



def _scalar_tensor_value(sample: Dict[str, Any], key: str, default: float = 0.0) -> torch.Tensor:
    if key not in sample:
        return torch.tensor([float(default)], dtype=torch.float32)
    x = sample[key]
    if isinstance(x, torch.Tensor):
        return x.float().reshape(-1)[:1]
    return torch.tensor([float(x)], dtype=torch.float32)


def _normalize_scalar_by_sample_bounds(
    sample: Dict[str, Any],
    *,
    value_key: str,
    min_key: str,
    max_key: str,
    default: float = 0.0,
) -> torch.Tensor:
    """
    Normalize one scalar to approximately [-1, 1] if min/max bounds exist.

    If no bounds are available, return default. This keeps old constant-material
    samples usable while allowing future samples with variable constant E/nu to
    provide E_min/E_max and nu_min/nu_max.
    """
    if value_key not in sample or min_key not in sample or max_key not in sample:
        return torch.tensor([float(default)], dtype=torch.float32)

    v = _scalar_tensor_value(sample, value_key, default=0.0)
    vmin = _scalar_tensor_value(sample, min_key, default=float(v.item()))
    vmax = _scalar_tensor_value(sample, max_key, default=float(v.item()))
    mid = 0.5 * (vmin + vmax)
    half = torch.clamp(0.5 * (vmax - vmin), min=1.0e-12)
    return ((v - mid) / half).float().reshape(1)


def _infer_load_scalar_from_sample(sample: Dict[str, Any]) -> torch.Tensor:
    """
    Return the normalized scalar pressure input with shape [1].

    This legacy helper is kept for compatibility. The new scalar parameter
    vector is sample["scalar_param"] = [E_net, nu_net, p_net].
    """
    if "load_scalar" in sample:
        x = sample["load_scalar"]
        if isinstance(x, torch.Tensor):
            return x.float().reshape(-1)[:1]
        return torch.tensor([float(x)], dtype=torch.float32)

    if "pressure_net" in sample:
        p = sample["pressure_net"]
        if isinstance(p, torch.Tensor):
            return p.float().mean().reshape(1)
        return torch.as_tensor(p, dtype=torch.float32).mean().reshape(1)

    if "pressure" in sample:
        p = sample["pressure"]
        if isinstance(p, torch.Tensor):
            return p.float().mean().reshape(1)
        return torch.tensor([float(p)], dtype=torch.float32)

    return torch.zeros(1, dtype=torch.float32)


def _infer_scalar_param_from_sample(sample: Dict[str, Any]) -> torch.Tensor:
    """
    Return normalized scalar parameters with shape [3]:
        [E_scalar_net, nu_scalar_net, p_scalar_net]

    Constant parameters should enter the network through this cheap MLP branch.
    Field parameters should additionally enter through material_pc_net or
    load_pc_net when use_material_field / use_load_field is true.
    """
    if "scalar_param" in sample:
        x = _as_tensor(sample["scalar_param"]).float().reshape(-1)
        if x.numel() < 3:
            x = torch.cat([x, torch.zeros(3 - x.numel(), dtype=x.dtype)], dim=0)
        return x[:3]

    E_net = _normalize_scalar_by_sample_bounds(
        sample,
        value_key="E",
        min_key="E_min",
        max_key="E_max",
        default=0.0,
    )
    nu_net = _normalize_scalar_by_sample_bounds(
        sample,
        value_key="nu",
        min_key="nu_min",
        max_key="nu_max",
        default=0.0,
    )
    p_net = _infer_load_scalar_from_sample(sample)
    return torch.cat([E_net, nu_net, p_net], dim=0).float()


def _has_material_field(sample: Dict[str, Any]) -> bool:
    if "use_material_field" in sample:
        return bool(sample["use_material_field"])
    return ("material_pc_net" in sample) or ("E_elem" in sample and "nu_elem" in sample)


def _has_load_field(sample: Dict[str, Any]) -> bool:
    if "use_load_field" in sample:
        return bool(sample["use_load_field"])
    if "pressure_net" in sample or "load_field" in sample:
        return True
    # Backward compatibility: old pressure-field samples used load_pc_net with
    # four channels [x,y,z,p]. Old L-shape material-field samples may also have
    # load_pc_net, but material_pc_net is present in those cases.
    if "load_pc_net" in sample and "material_pc_net" not in sample:
        try:
            return int(_as_tensor(sample["load_pc_net"]).shape[-1]) in (4, 5)
        except Exception:
            return True
    return False


def _ensure_load_field_pc(x: torch.Tensor) -> torch.Tensor:
    """
    Ensure a pressure/load field point cloud has four channels:
        [x_net, y_net, z_net, p_net]

    If an old five-channel array [x,y,z,p,0] is supplied, the last zero channel
    is dropped.
    """
    x = x.float()
    if x.ndim != 2:
        raise ValueError(f"Expected load field point cloud [N,C], got {tuple(x.shape)}")
    if x.shape[1] == 4:
        return x
    if x.shape[1] >= 5:
        return x[:, [0, 1, 2, 3]]
    raise ValueError(
        f"Expected load field point cloud with at least 4 channels, got {tuple(x.shape)}."
    )


def _ensure_material_field_pc(x: torch.Tensor) -> torch.Tensor:
    """
    Ensure a material field point cloud has five channels:
        [x_net, y_net, z_net, E_net, nu_net]
    """
    x = x.float()
    if x.ndim != 2:
        raise ValueError(f"Expected material field point cloud [N,C], got {tuple(x.shape)}")
    if x.shape[1] == 5:
        return x
    if x.shape[1] > 5:
        return x[:, :5]
    raise ValueError(
        f"Expected material field point cloud with 5 channels, got {tuple(x.shape)}."
    )


def _infer_load_pc_net_from_sample(sample: Dict[str, Any]) -> torch.Tensor:
    """
    Return normalized pressure/load field point cloud [N, 4]:
        [x_net, y_net, z_net, p_net]

    If the sample has no spatial load field, return one dummy zero point. The
    load-field encoder will be skipped when use_load_field is false.
    """
    if not _has_load_field(sample):
        return torch.zeros((1, 4), dtype=torch.float32)

    if "boundary_load_pc_net" in sample:
        return _ensure_load_field_pc(_as_tensor(sample["boundary_load_pc_net"]))

    if "load_pc_net" in sample and "material_pc_net" not in sample:
        return _ensure_load_field_pc(_as_tensor(sample["load_pc_net"]))

    query_net = _infer_query_net_from_sample(sample)

    if "pressure_net" in sample:
        p_net = _as_tensor(sample["pressure_net"]).reshape(query_net.shape[0], -1)[:, :1]
    elif "load_field" in sample:
        p_net = _as_tensor(sample["load_field"]).reshape(query_net.shape[0], -1)[:, :1]
    else:
        p_scalar = _infer_load_scalar_from_sample(sample).reshape(1, 1)
        p_net = p_scalar.expand(query_net.shape[0], 1).clone()

    return torch.cat([query_net[:, :3], p_net], dim=-1).float()


def _infer_material_pc_net_from_sample(sample: Dict[str, Any]) -> torch.Tensor:
    """
    Return normalized material field point cloud [N, 5]:
        [x_net, y_net, z_net, E_net, nu_net]

    If the sample has no material field, return one dummy zero point. The
    material-field encoder will be skipped when use_material_field is false.
    """
    if not _has_material_field(sample):
        return torch.zeros((1, 5), dtype=torch.float32)

    if "material_pc_net" in sample:
        return _ensure_material_field_pc(_as_tensor(sample["material_pc_net"]))

    raise KeyError(
        "Sample declares a material field but does not contain 'material_pc_net'."
    )


def _infer_bc_factor_from_sample(
    sample: Dict[str, Any],
    *,
    out_channels: int = 3,
) -> torch.Tensor:
    """
    Return a boundary-condition factor with shape [Nq, out_channels].

    Meaning:
        1.0 = unconstrained DOF
        0.0 = constrained DOF

    For the current MR plate:
        out_channels = 3
        DOFs = [w, theta_x, theta_y]

    Priority:
        1. sample["bc_factor"]
        2. sample["bc_clamped"] converted to [0,0,0] on clamped nodes
        3. all ones
    """
    query = _infer_query_from_sample(sample)
    n_q = int(query.shape[0])

    if "bc_factor" in sample:
        bc = _as_tensor(sample["bc_factor"])
        if bc.ndim == 1:
            bc = bc.view(-1, 1).expand(-1, out_channels)
        elif bc.ndim == 2 and bc.shape[1] == 1:
            bc = bc.expand(-1, out_channels)
        elif bc.ndim == 2 and bc.shape[1] >= out_channels:
            bc = bc[:, :out_channels]
        else:
            raise ValueError(
                f"Invalid bc_factor shape {tuple(bc.shape)}. "
                f"Expected [N], [N,1], or [N,{out_channels}]."
            )

        if bc.shape[0] != n_q:
            raise ValueError(
                f"bc_factor node count {bc.shape[0]} does not match "
                f"query node count {n_q}."
            )
        return bc.float()

    if "bc_clamped" in sample:
        clamped = sample["bc_clamped"]
        if isinstance(clamped, torch.Tensor):
            clamped_t = clamped.bool().reshape(-1)
        else:
            clamped_t = torch.as_tensor(clamped, dtype=torch.bool).reshape(-1)

        if clamped_t.shape[0] != n_q:
            raise ValueError(
                f"bc_clamped node count {clamped_t.shape[0]} does not match "
                f"query node count {n_q}."
            )

        bc = torch.ones((n_q, out_channels), dtype=torch.float32)
        bc[clamped_t, :] = 0.0
        return bc

    return torch.ones((n_q, out_channels), dtype=torch.float32)


def _infer_bbox_from_sample(sample: Dict[str, Any]) -> torch.Tensor:
    """
    bbox format:
        [x_min, x_max, y_min, y_max]
    """
    if "bbox" in sample:
        return _as_tensor(sample["bbox"]).reshape(4)

    query = _infer_query_from_sample(sample)
    x = query[:, 0]
    y = query[:, 1]
    return torch.tensor(
        [
            float(torch.min(x).item()),
            float(torch.max(x).item()),
            float(torch.min(y).item()),
            float(torch.max(y).item()),
        ],
        dtype=torch.float32,
    )


def plate_collate_fn(
    batch: List[Dict[str, Any]],
    *,
    padding_value: float = -999.0,
) -> Dict[str, Any]:
    """
    Collate variable-size MR-plate samples.

    Padded fields:
        pc, pc_net, query, query_net, pressure, bc_factor,
        load_pc_net, material_pc_net

    List fields:
        verts_xy, faces, geom_face

    Scalar fields:
        E, nu, thickness, load_scalar, scalar_param, lx, ly, bbox
    """
    out: Dict[str, Any] = {}

    pc_list = [_as_tensor(s["pc"]) for s in batch]
    pc_net_list = [_infer_pc_net_from_sample(s) for s in batch]
    query_list = [_infer_query_from_sample(s) for s in batch]
    query_net_list = [_infer_query_net_from_sample(s) for s in batch]

    out["pc"] = _pad_float_list(pc_list, padding_value=padding_value)
    out["pc_net"] = _pad_float_list(pc_net_list, padding_value=padding_value)
    out["query"] = _pad_float_list(query_list, padding_value=padding_value)
    out["query_net"] = _pad_float_list(query_net_list, padding_value=padding_value)

    out["load_pc_net"] = _pad_float_list(
        [_infer_load_pc_net_from_sample(s) for s in batch],
        padding_value=padding_value,
    )

    out["material_pc_net"] = _pad_float_list(
        [_infer_material_pc_net_from_sample(s) for s in batch],
        padding_value=padding_value,
    )

    out["use_load_field"] = torch.as_tensor(
        [_has_load_field(s) for s in batch],
        dtype=torch.bool,
    )
    out["use_material_field"] = torch.as_tensor(
        [_has_material_field(s) for s in batch],
        dtype=torch.bool,
    )

    out["pressure"] = _pad_float_list(
        [_as_tensor(s["pressure"]) for s in batch],
        padding_value=0.0,
    )

    out["bc_factor"] = _pad_float_list(
        [_infer_bc_factor_from_sample(s, out_channels=3) for s in batch],
        padding_value=0.0,
    )

    # Mesh data are kept as lists because different geometries may have
    # different node/element counts.
    out["verts_xy"] = [_as_tensor(s["verts_xy"]) for s in batch]
    out["faces"] = [torch.as_tensor(s["faces"], dtype=torch.long) for s in batch]
    out["geom_face"] = [s.get("geom_face", None) for s in batch]
    out["boundary_load_edges"] = [
        torch.as_tensor(s["loaded_boundary_edges"], dtype=torch.long)
        if "loaded_boundary_edges" in s
        else None
        for s in batch
    ]

    # Elementwise material fields are also kept as lists because each sample may
    # have a different number of triangular elements.
    out["E_elem"] = [
        _as_tensor(s["E_elem"]).float().reshape(-1) if "E_elem" in s else None
        for s in batch
    ]

    out["nu_elem"] = [
        _as_tensor(s["nu_elem"]).float().reshape(-1) if "nu_elem" in s else None
        for s in batch
    ]

    # Scalars.
    out["E"] = torch.stack([_as_tensor(s.get("E", [1.0])).reshape(-1) for s in batch], dim=0)
    out["nu"] = torch.stack([_as_tensor(s.get("nu", [0.3])).reshape(-1) for s in batch], dim=0)
    out["thickness"] = torch.stack(
        [_as_tensor(s.get("thickness", [1.0])).reshape(-1) for s in batch],
        dim=0,
    )

    out["load_scalar"] = torch.stack(
        [_infer_load_scalar_from_sample(s) for s in batch],
        dim=0,
    )

    out["scalar_param"] = torch.stack(
        [_infer_scalar_param_from_sample(s) for s in batch],
        dim=0,
    )

    out["sample_ids"] = torch.as_tensor(
        [int(s.get("sample_id", i)) for i, s in enumerate(batch)],
        dtype=torch.long,
    )

    out["lx"] = torch.as_tensor([float(s.get("lx", 1.0)) for s in batch], dtype=torch.float32)
    out["ly"] = torch.as_tensor([float(s.get("ly", 1.0)) for s in batch], dtype=torch.float32)
    out["bbox"] = torch.stack([_infer_bbox_from_sample(s) for s in batch], dim=0)

    out["geometry_type"] = [str(s.get("geometry_type", "")) for s in batch]
    out["load_type"] = [str(s.get("load_type", "")) for s in batch]

    return out


# -----------------------------------------------------------------------------
# GINOT-like MR plate operator
# -----------------------------------------------------------------------------

def mr_plate_ginot_minimal_configs() -> Dict[str, Any]:
    """
    Minimal GINOT-like configuration for the MR plate model.

    The model uses one geometry encoder, one material-field encoder, one
    load-field encoder, and one scalar-parameter encoder. Field encoders are
    skipped at runtime when the corresponding field is absent in a batch.
    """
    padding_value = -999.0
    out_c = 64  # attention embedding dimension

    # Geometry Encoder: boundary point cloud [x,y,z].
    branch_args = {
        "input_channels": 3,
        "out_c": out_c,
        "latent_d": None,
        "width": 64,
        "n_point": 256,
        "n_sample": 16,
        "radius": 0.20,
        "d_hidden": [64, 64],
        "num_heads": 4,
        "cross_attn_layers": 1,
        "self_attn_layers": 2,
        "fps_method": "fps",
        "pc_padding_val": padding_value,
        "dropout": 0.0,
    }

    # Material-field encoder: [x,y,z,E_net,nu_net].
    material_branch_args = dict(branch_args)
    material_branch_args["input_channels"] = 5

    # Load-field encoder: [x,y,z,p_net].
    load_branch_args = dict(branch_args)
    load_branch_args["input_channels"] = 4

    # Solution Decoder.
    trunk_args = {
        "embed_dim": out_c,
        "num_heads": 4,
        "cross_attn_layers": 4,
        "in_channels": 3,
        "out_channels": 3,
        "use_nerf_query": True,
        "out_mean": [0.0, 0.0, 0.0],
        "out_std": [2.0e-3, 5.0e-3, 5.0e-3],
        "dropout": 0.0,
        "padding_value": padding_value,
        "scalar_dim": out_c,
    }

    return {
        "padding_value": padding_value,
        "branch_args": branch_args,
        "material_branch_args": material_branch_args,
        "load_branch_args": load_branch_args,
        "trunk_args": trunk_args,
        "hidden_dim": out_c,
        "latent_dim": out_c,
        "num_layers": 4,
    }


class PlateGINOTQueryDecoder(nn.Module):
    """
    GINOT-like encoder-decoder for Mindlin-Reissner plate response.

    Inputs from batch:
        pc_net          : [B, Ng, 3], normalized boundary point cloud
        query_net       : [B, Nq, 3], normalized query coordinates
        material_pc_net : [B, Nm, 5], [x,y,z,E_net,nu_net]
        load_pc_net     : [B, Nl, 4], [x,y,z,p_net]
        scalar_param    : [B, 3], [E_scalar_net, nu_scalar_net, p_scalar_net]

    Field encoders are skipped when use_material_field/use_load_field is false.
    Fusion still uses feature concatenation, so all latent streams are truncated
    to a common token count before concatenation.

    Output:
        [B, Nq, 3] = [w, theta_x, theta_y] in physical scale.
    """

    def __init__(
        self,
        branch: nn.Module,
        material_branch: Optional[nn.Module] = None,
        load_branch: Optional[nn.Module] = None,
        *,
        embed_dim: int = 64,
        geom_dim: int = 64,
        material_dim: int = 64,
        load_dim: int = 64,
        scalar_dim: int = 64,
        num_heads: int = 4,
        cross_attn_layers: int = 4,
        in_channels: int = 3,
        out_channels: int = 3,
        use_nerf_query: bool = True,
        out_mean: Optional[Sequence[float]] = None,
        out_std: Optional[Sequence[float]] = None,
        dropout: float = 0.0,
        padding_value: float = -999.0,
    ):
        super().__init__()
        self.branch = branch
        self.material_branch = material_branch
        self.load_branch = load_branch
        self.padding_value = float(padding_value)
        self.out_channels = int(out_channels)
        self.embed_dim = int(embed_dim)
        self.geom_dim = int(geom_dim)
        self.material_dim = int(material_dim)
        self.load_dim = int(load_dim)
        self.scalar_dim = int(scalar_dim)

        if use_nerf_query:
            self.q_linear = PosEmbLinear("nerf", in_channels, embed_dim)
        else:
            self.q_linear = nn.Linear(in_channels, embed_dim)
        self.q_mlp = MLP(width=embed_dim)

        self.scalar_param_encoder = nn.Sequential(
            nn.Linear(3, 2 * scalar_dim),
            nn.SiLU(),
            nn.Linear(2 * scalar_dim, 2 * scalar_dim),
            nn.SiLU(),
            nn.Linear(2 * scalar_dim, scalar_dim),
            nn.SiLU(),
        )

        # Backward-compatible alias for older scripts/checkpoint inspection.
        self.loading_encoder = self.scalar_param_encoder

        fusion_in_dim = int(geom_dim) + int(material_dim) + int(load_dim) + int(scalar_dim)

        self.latent_encoder = nn.Sequential(
            nn.Linear(fusion_in_dim, 2 * embed_dim),
            nn.SiLU(),
            nn.Linear(2 * embed_dim, embed_dim),
            nn.SiLU(),
        )

        self.resblocks = nn.ModuleList(
            [
                ResidualCrossAttentionBlock(
                    width=embed_dim,
                    heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(cross_attn_layers)
            ]
        )

        self.output_proj = nn.Linear(embed_dim, out_channels)

        if out_mean is None:
            out_mean = [0.0] * out_channels
        if out_std is None:
            out_std = [2.0e-3, 5.0e-3, 5.0e-3]

        self.register_buffer("out_mean", torch.tensor(out_mean, dtype=torch.float32))
        self.register_buffer("out_std", torch.tensor(out_std, dtype=torch.float32))

        nn.init.normal_(self.output_proj.weight, mean=0.0, std=1.0e-4)
        nn.init.zeros_(self.output_proj.bias)

    def decode_output(self, y_norm: torch.Tensor) -> torch.Tensor:
        return (
            self.out_mean.to(device=y_norm.device, dtype=y_norm.dtype)
            + y_norm * self.out_std.to(device=y_norm.device, dtype=y_norm.dtype)
        )

    @staticmethod
    def _flag_any(batch: Dict[str, Any], key: str) -> bool:
        v = batch.get(key, False)
        if isinstance(v, torch.Tensor):
            return bool(v.detach().cpu().bool().any().item())
        if isinstance(v, (list, tuple)):
            return any(bool(x) for x in v)
        return bool(v)

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        query_net = batch["query_net"].to(device)
        pc_net = batch["pc_net"].to(device)[..., :3]
        material_pc_net = batch["material_pc_net"].to(device)
        load_pc_net = batch["load_pc_net"].to(device)
        scalar_param = batch.get("scalar_param", None)
        if scalar_param is None:
            # Legacy fallback: [0,0,load_scalar].
            load_scalar = batch["load_scalar"].to(device)
            if load_scalar.ndim == 1:
                load_scalar = load_scalar.view(load_scalar.shape[0], 1)
            scalar_param = torch.cat(
                [torch.zeros_like(load_scalar), torch.zeros_like(load_scalar), load_scalar[:, :1]],
                dim=-1,
            )
        else:
            scalar_param = scalar_param.to(device)

        sample_ids = batch.get("sample_ids", None)
        if sample_ids is not None:
            sample_ids = sample_ids.to(device)

        # Geometry encoder: [B, Ng, 3] -> [B, Nlatent, C].
        geom_latent = self.branch(
            pc_net,
            sample_ids=sample_ids,
            apply_padding_pointnet2=True,
        )

        B, n_latent, _ = geom_latent.shape

        # Material field encoder: [B, Nm, 5] -> [B, Nmat, C].
        if self.material_branch is not None and self._flag_any(batch, "use_material_field"):
            material_latent = self.material_branch(
                material_pc_net,
                sample_ids=None,
                apply_padding_pointnet2=True,
            )
        else:
            material_latent = torch.zeros(
                (B, n_latent, self.material_dim),
                dtype=geom_latent.dtype,
                device=geom_latent.device,
            )

        # Load/pressure field encoder: [B, Nl, 4] -> [B, Nload, C].
        if self.load_branch is not None and self._flag_any(batch, "use_load_field"):
            load_latent = self.load_branch(
                load_pc_net,
                sample_ids=None,
                apply_padding_pointnet2=True,
            )
        else:
            load_latent = torch.zeros(
                (B, n_latent, self.load_dim),
                dtype=geom_latent.dtype,
                device=geom_latent.device,
            )

        n_common = min(
            int(geom_latent.shape[1]),
            int(material_latent.shape[1]),
            int(load_latent.shape[1]),
        )
        geom_latent = geom_latent[:, :n_common]
        material_latent = material_latent[:, :n_common]
        load_latent = load_latent[:, :n_common]
        n_latent = n_common

        if geom_latent.shape[-1] != self.geom_dim:
            raise RuntimeError(
                f"Geometry encoder output dim mismatch: "
                f"got {geom_latent.shape[-1]}, expected {self.geom_dim}."
            )
        if material_latent.shape[-1] != self.material_dim:
            raise RuntimeError(
                f"Material-field encoder output dim mismatch: "
                f"got {material_latent.shape[-1]}, expected {self.material_dim}."
            )
        if load_latent.shape[-1] != self.load_dim:
            raise RuntimeError(
                f"Load-field encoder output dim mismatch: "
                f"got {load_latent.shape[-1]}, expected {self.load_dim}."
            )

        if scalar_param.ndim == 1:
            scalar_param = scalar_param.view(B, -1)
        if scalar_param.shape[1] < 3:
            pad = torch.zeros((B, 3 - scalar_param.shape[1]), dtype=scalar_param.dtype, device=scalar_param.device)
            scalar_param = torch.cat([scalar_param, pad], dim=-1)
        scalar_param = scalar_param[:, :3]

        scalar_latent = self.scalar_param_encoder(scalar_param)
        scalar_latent = scalar_latent.view(B, 1, -1).expand(-1, n_latent, -1)

        latent = torch.cat([geom_latent, material_latent, load_latent, scalar_latent], dim=-1)
        latent = self.latent_encoder(latent)

        q = self.q_mlp(self.q_linear(query_net))
        for block in self.resblocks:
            q = block(q, latent)

        return self.decode_output(self.output_proj(q))


def build_plate_operator_model(
    *,
    padding_value: float = -999.0,
    hidden_dim: int = 128,
    latent_dim: int = 128,
    num_layers: int = 4,
    model_config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> PlateGINOTQueryDecoder:
    """
    Build the GINOT-like MR plate operator model.

    The legacy arguments hidden_dim/latent_dim/num_layers are kept only for
    compatibility with the existing trainer.py. The actual architecture is
    controlled by mr_plate_ginot_minimal_configs() or model_config.
    """
    cfg = deepcopy(mr_plate_ginot_minimal_configs())

    material_branch_overridden = False
    load_branch_overridden = False
    if model_config is not None:
        for k, v in model_config.items():
            if k in ("branch_args", "material_branch_args", "load_branch_args", "trunk_args") and isinstance(v, dict):
                cfg[k].update(v)
                if k == "material_branch_args":
                    material_branch_overridden = True
                if k == "load_branch_args":
                    load_branch_overridden = True
            else:
                cfg[k] = v

    if not material_branch_overridden:
        cfg["material_branch_args"] = dict(cfg["branch_args"])
        cfg["material_branch_args"]["input_channels"] = 5

    if not load_branch_overridden:
        cfg["load_branch_args"] = dict(cfg["branch_args"])
        cfg["load_branch_args"]["input_channels"] = 4

    cfg["padding_value"] = float(padding_value)
    cfg["branch_args"]["pc_padding_val"] = float(padding_value)
    cfg["material_branch_args"]["pc_padding_val"] = float(padding_value)
    cfg["load_branch_args"]["pc_padding_val"] = float(padding_value)
    cfg["trunk_args"]["padding_value"] = float(padding_value)

    geom_dim = int(cfg["branch_args"].get("out_c", cfg["trunk_args"].get("embed_dim", 64)))
    material_dim = int(cfg["material_branch_args"].get("out_c", cfg["trunk_args"].get("embed_dim", 64)))
    load_dim = int(cfg["load_branch_args"].get("out_c", cfg["trunk_args"].get("embed_dim", 64)))

    trunk_args = dict(cfg["trunk_args"])
    embed_dim = int(trunk_args.get("embed_dim", geom_dim))
    scalar_dim = int(trunk_args.pop("scalar_dim", embed_dim))

    branch = PointCloudPerceiverChannelsEncoder(**cfg["branch_args"])
    material_branch = PointCloudPerceiverChannelsEncoder(**cfg["material_branch_args"])
    load_branch = PointCloudPerceiverChannelsEncoder(**cfg["load_branch_args"])

    model = PlateGINOTQueryDecoder(
        branch=branch,
        material_branch=material_branch,
        load_branch=load_branch,
        geom_dim=geom_dim,
        material_dim=material_dim,
        load_dim=load_dim,
        scalar_dim=scalar_dim,
        **trunk_args,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Trainable parameters of MR plate GINOT-like operator: {n_params}")

    return model


def triangle_xy_area_and_grads(
    verts_xy: torch.Tensor,
    faces: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tri = verts_xy[faces]

    x1 = tri[:, 0, 0]
    y1 = tri[:, 0, 1]
    x2 = tri[:, 1, 0]
    y2 = tri[:, 1, 1]
    x3 = tri[:, 2, 0]
    y3 = tri[:, 2, 1]

    two_area_signed = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    two_area = torch.abs(two_area_signed)
    area = 0.5 * two_area

    denom = torch.where(
        torch.abs(two_area_signed) > 1.0e-14,
        two_area_signed,
        torch.full_like(two_area_signed, 1.0e-14),
    )

    dNdx = torch.stack(
        [
            (y2 - y3) / denom,
            (y3 - y1) / denom,
            (y1 - y2) / denom,
        ],
        dim=1,
    )

    dNdy = torch.stack(
        [
            (x3 - x2) / denom,
            (x1 - x3) / denom,
            (x2 - x1) / denom,
        ],
        dim=1,
    )

    return area, dNdx, dNdy


def plate_isotropic_lame_plane_stress(E: torch.Tensor, nu: torch.Tensor):
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / (1.0 - nu * nu)
    return lam, mu


def _as_geom_tensor(
    geom_face: Optional[Dict[str, torch.Tensor]],
    key: str,
    *,
    fallback: Optional[torch.Tensor] = None,
    device_: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if geom_face is not None and key in geom_face:
        val = geom_face[key]
        if isinstance(val, torch.Tensor):
            return val.to(device=device_, dtype=dtype)
        return torch.as_tensor(val, device=device_, dtype=dtype)
    if fallback is None:
        raise KeyError(f"Missing geometry key '{key}' and no fallback was provided.")
    return fallback.to(device=device_, dtype=dtype)


def mr_plate_total_potential_one_sample(
    *,
    y_pred_local: torch.Tensor,
    verts_xy: torch.Tensor,
    faces: torch.Tensor,
    pressure_nodes: torch.Tensor,
    E: torch.Tensor,
    nu: torch.Tensor,
    thickness: torch.Tensor,
    E_elem: Optional[torch.Tensor] = None,
    nu_elem: Optional[torch.Tensor] = None,
    shear_correction: float = 5.0 / 6.0,
    include_shear_energy: bool = True,
    geom_face: Optional[Dict[str, torch.Tensor]] = None,
    boundary_load_edges: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute NO-side linear Mindlin-Reissner plate total potential.

    y_pred_local:
        [N, 3] = [w, theta_x, theta_y]

    pressure_nodes:
        [N] or [N, 1]. For a boundary line load, values are zero in the
        domain and nonzero at nodes belonging to boundary_load_edges.
    """
    verts_xy = verts_xy.to(device)
    faces = faces.to(device)
    y_pred_local = y_pred_local.to(device)
    pressure_nodes = pressure_nodes.to(device).reshape(-1)

    dtype = y_pred_local.dtype
    E = E.to(device=device, dtype=dtype).reshape(())
    nu = nu.to(device=device, dtype=dtype).reshape(())
    thickness = thickness.to(device=device, dtype=dtype).reshape(())

    n_elem = int(faces.shape[0])

    # Effective material used by the NO-side energy.
    # If E_elem/nu_elem are provided, use one material value per element.
    # Otherwise fall back to uniform scalar E/nu for old uniform-material cases.
    if (E_elem is None) != (nu_elem is None):
        raise ValueError(
            "E_elem and nu_elem must be provided together. "
            f"Got E_elem is None: {E_elem is None}, nu_elem is None: {nu_elem is None}."
        )

    if E_elem is not None and nu_elem is not None:
        E_eff = E_elem.to(device=device, dtype=dtype).reshape(-1)
        nu_eff = nu_elem.to(device=device, dtype=dtype).reshape(-1)

        if E_eff.numel() != n_elem or nu_eff.numel() != n_elem:
            raise ValueError(
                f"E_elem/nu_elem must have one value per triangular face. "
                f"Got E_elem={E_eff.numel()}, nu_elem={nu_eff.numel()}, n_elem={n_elem}."
            )
    else:
        E_eff = E.expand(n_elem)
        nu_eff = nu.expand(n_elem)

    if geom_face is None:
        area, dNdx, dNdy = triangle_xy_area_and_grads(verts_xy, faces)
        area = area.to(device=device, dtype=dtype)
        dNdx = dNdx.to(device=device, dtype=dtype)
        dNdy = dNdy.to(device=device, dtype=dtype)

        base_mass = torch.tensor(
            [
                [2.0, 1.0, 1.0],
                [1.0, 2.0, 1.0],
                [1.0, 1.0, 2.0],
            ],
            dtype=dtype,
            device=device,
        )
        mass_N = area[:, None, None] / 12.0 * base_mass[None, :, :]
        int_N = area[:, None] / 3.0
    else:
        area_fallback, dNdx_fallback, dNdy_fallback = triangle_xy_area_and_grads(verts_xy, faces)
        area = _as_geom_tensor(
            geom_face,
            "area_phys" if "area_phys" in geom_face else "area",
            fallback=area_fallback,
            device_=device,
            dtype=dtype,
        )
        dNdx = _as_geom_tensor(geom_face, "dNdx", fallback=dNdx_fallback, device_=device, dtype=dtype)
        dNdy = _as_geom_tensor(geom_face, "dNdy", fallback=dNdy_fallback, device_=device, dtype=dtype)

        if "mass_N" in geom_face:
            mass_N = _as_geom_tensor(geom_face, "mass_N", device_=device, dtype=dtype)
        else:
            base_mass = torch.tensor(
                [
                    [2.0, 1.0, 1.0],
                    [1.0, 2.0, 1.0],
                    [1.0, 1.0, 2.0],
                ],
                dtype=dtype,
                device=device,
            )
            mass_N = area[:, None, None] / 12.0 * base_mass[None, :, :]

        if "int_N" in geom_face:
            int_N = _as_geom_tensor(geom_face, "int_N", device_=device, dtype=dtype)
        else:
            int_N = area[:, None] / 3.0

        # Some old files store int_N as [E, 1]. Expand it to [E, 3].
        if int_N.ndim == 2 and int_N.shape[1] == 1:
            int_N = int_N.expand(-1, 3)

    tri_q = y_pred_local[faces]

    w_nodes = tri_q[:, :, 0]
    thx_nodes = tri_q[:, :, 1]
    thy_nodes = tri_q[:, :, 2]
    theta_nodes = tri_q[:, :, 1:3]

    dw_dx = torch.einsum("ei,ei->e", dNdx, w_nodes)
    dw_dy = torch.einsum("ei,ei->e", dNdy, w_nodes)

    dthx_dx = torch.einsum("ei,ei->e", dNdx, thx_nodes)
    dthx_dy = torch.einsum("ei,ei->e", dNdy, thx_nodes)
    dthy_dx = torch.einsum("ei,ei->e", dNdx, thy_nodes)
    dthy_dy = torch.einsum("ei,ei->e", dNdy, thy_nodes)

    kxx = dthx_dx
    kyy = dthy_dy
    kxy = 0.5 * (dthx_dy + dthy_dx)

    tr_kappa = kxx + kyy
    kappa_contract = kxx * kxx + kyy * kyy + 2.0 * kxy * kxy

    lam, mu = plate_isotropic_lame_plane_stress(E_eff, nu_eff)

    U_ben_e = 0.5 * (thickness**3 / 12.0) * (
        lam * area * tr_kappa * tr_kappa
        + 2.0 * mu * area * kappa_contract
    )

    grad_w_sq = dw_dx * dw_dx + dw_dy * dw_dy

    theta_int = torch.einsum("ei,eia->ea", int_N, theta_nodes)
    grad_w_theta_int = dw_dx * theta_int[:, 0] + dw_dy * theta_int[:, 1]

    theta_quad = torch.einsum(
        "eij,eia,eja->e",
        mass_N,
        theta_nodes,
        theta_nodes,
    )

    # Exact P1-element integral of ||theta - grad(w)||^2.  It is retained as a
    # diagnostic for the Mindlin-Reissner shear constraint.
    constraint_sq_int_e = (
        area * grad_w_sq - 2.0 * grad_w_theta_int + theta_quad
    )

    if include_shear_energy:
        U_shr_e = 0.5 * shear_correction * mu * thickness * (
            constraint_sq_int_e
        )
    else:
        # Kirchhoff-Love thin-plate model: transverse shear deformation is
        # neglected, hence it makes no contribution to the potential energy.
        U_shr_e = torch.zeros_like(U_ben_e)

    if boundary_load_edges is None:
        p_nodes = pressure_nodes[faces]
        W_ext_e = torch.einsum("eij,ei,ej->e", mass_N, p_nodes, w_nodes)
    else:
        edges = boundary_load_edges.to(device=device, dtype=torch.long)
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError(
                "boundary_load_edges must have shape [n_edges, 2], "
                f"got {tuple(edges.shape)}."
            )
        edge_xy = verts_xy[edges]
        edge_length = torch.linalg.vector_norm(edge_xy[:, 1] - edge_xy[:, 0], dim=1)
        q_edge = pressure_nodes[edges]
        w_edge = y_pred_local[edges, 0]
        # Exact integral of the product of two linear fields on each edge.
        W_ext_e = edge_length / 6.0 * (
            2.0 * q_edge[:, 0] * w_edge[:, 0]
            + q_edge[:, 0] * w_edge[:, 1]
            + q_edge[:, 1] * w_edge[:, 0]
            + 2.0 * q_edge[:, 1] * w_edge[:, 1]
        )

    U_ben = torch.sum(U_ben_e)
    U_shr = torch.sum(U_shr_e)
    U_mem = torch.zeros_like(U_ben)
    U_int = U_ben + U_shr
    W_ext = torch.sum(W_ext_e)
    Pi = U_int - W_ext

    return {
        "Pi": Pi,
        "U_int": U_int,
        "U_mem": U_mem,
        "U_ben": U_ben,
        "U_shr": U_shr,
        "constraint_l2": torch.sqrt(
            torch.clamp(torch.sum(constraint_sq_int_e) / torch.sum(area), min=0.0)
        ),
        "W_ext": W_ext,
    }


class MRPlateEnergyOperator(nn.Module):
    """
    Wrapper:
        GINOT-like neural operator prediction + configurable essential-boundary
        transform + energy loss.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        padding_value: float = -999.0,
        boundary_transform: str = "binary_factor",
        include_shear_energy: bool = True,
    ):
        super().__init__()
        self.model = model
        self.padding_value = float(padding_value)
        boundary_transform = str(boundary_transform).strip().lower()
        if boundary_transform not in {"binary_factor", "square_clamped_distance"}:
            raise ValueError(
                "boundary_transform must be 'binary_factor' or "
                f"'square_clamped_distance', got {boundary_transform!r}."
            )
        self.boundary_transform = boundary_transform
        self.include_shear_energy = bool(include_shear_energy)

    def apply_boundary_transform(
        self,
        raw: torch.Tensor,
        batch: Dict[str, Any],
    ) -> torch.Tensor:
        """Apply the configured essential-boundary output transform.

        ``square_clamped_distance`` is intended for a rectangular plate whose
        normalized query coordinates lie in ``[-1, 1]^2``.  The polynomial

            phi(xi, eta) = (1 - xi**2) (1 - eta**2)

        vanishes on all four outer edges and remains nonzero at an interior
        hole.  Multiplying all three Mindlin--Reissner unknowns by ``phi``
        enforces ``w = theta_x = theta_y = 0`` continuously.  Because ``phi``
        is built from ``query_net`` here, coordinate automatic differentiation
        includes its first and second derivatives.
        """
        if self.boundary_transform == "binary_factor":
            bc_factor = batch["bc_factor"].to(device=raw.device, dtype=raw.dtype)
            return raw * bc_factor

        query_net = batch["query_net"].to(device=raw.device, dtype=raw.dtype)
        xi = query_net[..., 0]
        eta = query_net[..., 1]
        phi = (1.0 - xi.square()) * (1.0 - eta.square())

        # Padded query coordinates use a large sentinel.  Do not let the
        # polynomial transform amplify those unused positions.
        query = batch["query"].to(device=raw.device)
        valid = query[..., 0] != self.padding_value
        phi = torch.where(valid, phi, torch.zeros_like(phi))
        return raw * phi.unsqueeze(-1)

    def predict(self, batch: Dict[str, Any]) -> torch.Tensor:
        raw = self.model(batch)
        return self.apply_boundary_transform(raw, batch)

    def forward(self, batch: Dict[str, Any]):
        y_eff = self.predict(batch)

        query = batch["query"].to(device)
        q_valid = query[:, :, 0] != self.padding_value

        energy_terms = []
        stats = {
            "Pi": [],
            "U_int": [],
            "U_mem": [],
            "U_ben": [],
            "U_shr": [],
            "constraint_l2": [],
            "W_ext": [],
        }

        B = query.shape[0]

        for b in range(B):
            n_q = int(q_valid[b].sum().item())

            y_b = y_eff[b, :n_q]
            p_b = batch["pressure"][b, :n_q].to(device)

            verts_xy = batch["verts_xy"][b].to(device)
            faces = batch["faces"][b].to(device)
            geom_face = None
            if "geom_face" in batch and batch["geom_face"][b] is not None:
                geom_face = batch["geom_face"][b]

            E = batch["E"][b, 0].to(device)
            nu = batch["nu"][b, 0].to(device)
            thickness = batch["thickness"][b, 0].to(device)

            E_elem = None
            nu_elem = None

            if "E_elem" in batch and batch["E_elem"][b] is not None:
                E_elem = batch["E_elem"][b].to(device)

            if "nu_elem" in batch and batch["nu_elem"][b] is not None:
                nu_elem = batch["nu_elem"][b].to(device)

            terms = mr_plate_total_potential_one_sample(
                y_pred_local=y_b,
                verts_xy=verts_xy,
                faces=faces,
                pressure_nodes=p_b,
                E=E,
                nu=nu,
                thickness=thickness,
                E_elem=E_elem,
                nu_elem=nu_elem,
                geom_face=geom_face,
                include_shear_energy=self.include_shear_energy,
                boundary_load_edges=(
                    batch["boundary_load_edges"][b]
                    if "boundary_load_edges" in batch
                    else None
                ),
            )

            energy_terms.append(terms["Pi"])

            for k in stats:
                stats[k].append(terms[k].detach())

        loss_energy = torch.stack(energy_terms).mean()

        stats_mean = {
            k: float(torch.stack(v).mean().detach().cpu().item())
            for k, v in stats.items()
        }

        return loss_energy, y_eff, stats_mean


class KirchhoffLovePlateEnergyOperator(nn.Module):
    """A displacement-only Kirchhoff--Love plate energy operator.

    The network predicts only the transverse displacement ``w``.  Rotations
    exported for compatibility are derived as ``grad(w)``; they are never
    independent trainable fields.  The bending energy uses the Hessian of
    ``w``, which enforces the KL kinematic relation and prevents the unbounded
    ``U_b(theta) - W(w)`` objective produced by merely dropping MR shear.

    This implementation is intentionally restricted to the clamped outer
    square used by the single-hole thickness cases.  The squared analytic
    factor enforces both ``w=0`` and ``grad(w)=0`` on all four outer edges.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        padding_value: float = -999.0,
    ):
        super().__init__()
        self.model = model
        self.padding_value = float(padding_value)

    def _field_and_derivatives(
        self,
        batch: Dict[str, Any],
        *,
        need_hessian: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return ``[w, w_x, w_y]`` and, when requested, ``[w_xx,w_yy,w_xy]``.

        Query coordinates are normalized independently for every sample.  The
        returned derivatives are converted back to physical x/y coordinates.
        ``torch.enable_grad`` keeps inference/export compatible with callers
        decorated by ``torch.no_grad``.
        """
        outer_grad_enabled = torch.is_grad_enabled()
        with torch.enable_grad():
            query_net = (
                batch["query_net"]
                .to(device=device)
                .detach()
                .clone()
                .requires_grad_(True)
            )
            model_batch = dict(batch)
            model_batch["query_net"] = query_net
            raw = self.model(model_batch)
            if raw.shape[-1] != 1:
                raise ValueError(
                    "KirchhoffLovePlateEnergyOperator requires a one-channel "
                    f"displacement model, got output shape {tuple(raw.shape)}."
                )

            # query_net is [-1,1]^2 for a hole plate.  Squaring the standard
            # boundary bubble makes w and its normal derivative vanish on the
            # complete outer boundary, while leaving the inner hole free.
            q_xy = query_net[..., :2].clamp(min=-1.0, max=1.0)
            bubble = ((1.0 - q_xy[..., 0] ** 2) * (1.0 - q_xy[..., 1] ** 2)) ** 2
            w = raw[..., 0] * bubble

            # A first derivative must retain its graph when a Hessian is
            # needed, irrespective of the outer grad mode.
            grad_q = torch.autograd.grad(
                w.sum(),
                query_net,
                create_graph=need_hessian,
                retain_graph=need_hessian,
            )[0][..., :2]

            bbox = batch["bbox"].to(device=device, dtype=w.dtype)
            sx = (2.0 / torch.clamp(bbox[:, 1] - bbox[:, 0], min=1.0e-12)).view(-1, 1)
            sy = (2.0 / torch.clamp(bbox[:, 3] - bbox[:, 2], min=1.0e-12)).view(-1, 1)
            theta_x = grad_q[..., 0] * sx
            theta_y = grad_q[..., 1] * sy
            prediction = torch.stack((w, theta_x, theta_y), dim=-1)

            hessian = None
            if need_hessian:
                # Symmetrize the two mixed partial evaluations to reduce small
                # automatic-differentiation roundoff asymmetries.
                grad_x_q = torch.autograd.grad(
                    grad_q[..., 0].sum(),
                    query_net,
                    create_graph=outer_grad_enabled,
                    retain_graph=True,
                )[0][..., :2]
                grad_y_q = torch.autograd.grad(
                    grad_q[..., 1].sum(),
                    query_net,
                    create_graph=outer_grad_enabled,
                    retain_graph=True,
                )[0][..., :2]
                hessian = torch.stack(
                    (
                        grad_x_q[..., 0] * sx * sx,
                        grad_y_q[..., 1] * sy * sy,
                        0.5 * (grad_x_q[..., 1] + grad_y_q[..., 0]) * sx * sy,
                    ),
                    dim=-1,
                )

        if not outer_grad_enabled:
            prediction = prediction.detach()
            if hessian is not None:
                hessian = hessian.detach()
        return prediction, hessian

    def predict(self, batch: Dict[str, Any]) -> torch.Tensor:
        prediction, _ = self._field_and_derivatives(batch, need_hessian=False)
        return prediction

    def forward(self, batch: Dict[str, Any]):
        prediction, hessian = self._field_and_derivatives(batch, need_hessian=True)
        assert hessian is not None

        query = batch["query"].to(device)
        q_valid = query[:, :, 0] != self.padding_value
        energy_terms = []
        # Keep the same statistics schema as the MR operator. Pure KL has no
        # transverse-shear term, so that entry is identically zero.
        stats = {
            k: []
            for k in (
                "Pi", "U_int", "U_mem", "U_ben", "U_shr",
                "constraint_l2", "W_ext",
            )
        }

        for b in range(query.shape[0]):
            n_q = int(q_valid[b].sum().item())
            verts_xy = batch["verts_xy"][b].to(device)
            faces = batch["faces"][b].to(device)
            geom_face = batch["geom_face"][b]
            dtype = prediction.dtype

            area_fallback, _, _ = triangle_xy_area_and_grads(verts_xy, faces)
            area = _as_geom_tensor(
                geom_face,
                "area_phys" if geom_face is not None and "area_phys" in geom_face else "area",
                fallback=area_fallback,
                device_=device,
                dtype=dtype,
            )
            if geom_face is not None and "mass_N" in geom_face:
                mass_N = _as_geom_tensor(
                    geom_face, "mass_N", device_=device, dtype=dtype
                )
            else:
                base_mass = torch.tensor(
                    [[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]],
                    dtype=dtype,
                    device=device,
                )
                mass_N = area[:, None, None] / 12.0 * base_mass[None, :, :]

            E = batch["E"][b, 0].to(device=device, dtype=dtype)
            nu = batch["nu"][b, 0].to(device=device, dtype=dtype)
            thickness = batch["thickness"][b, 0].to(device=device, dtype=dtype)
            n_elem = int(faces.shape[0])
            E_eff = E.expand(n_elem)
            nu_eff = nu.expand(n_elem)
            if batch.get("E_elem", [None])[b] is not None:
                E_eff = batch["E_elem"][b].to(device=device, dtype=dtype).reshape(-1)
                nu_eff = batch["nu_elem"][b].to(device=device, dtype=dtype).reshape(-1)

            # Vertex quadrature of the smooth neural Hessian is exact to first
            # order and avoids treating the KL displacement as a P1 field.
            h_face = hessian[b, :n_q][faces].mean(dim=1)
            kxx, kyy, kxy = h_face[:, 0], h_face[:, 1], h_face[:, 2]
            tr_kappa = kxx + kyy
            kappa_contract = kxx * kxx + kyy * kyy + 2.0 * kxy * kxy
            lam, mu = plate_isotropic_lame_plane_stress(E_eff, nu_eff)
            U_ben = torch.sum(
                0.5 * (thickness**3 / 12.0) * area * (
                    lam * tr_kappa * tr_kappa + 2.0 * mu * kappa_contract
                )
            )

            w_nodes = prediction[b, :n_q, 0][faces]
            pressure_nodes = batch["pressure"][b, :n_q, 0].to(device=device, dtype=dtype)
            boundary_edges = batch.get("boundary_load_edges", [None])[b]
            if boundary_edges is None:
                p_nodes = pressure_nodes[faces]
                W_ext = torch.sum(torch.einsum("eij,ei,ej->e", mass_N, p_nodes, w_nodes))
            else:
                edges = boundary_edges.to(device=device, dtype=torch.long)
                edge_xy = verts_xy[edges]
                edge_length = torch.linalg.vector_norm(edge_xy[:, 1] - edge_xy[:, 0], dim=1)
                q_edge = pressure_nodes[edges]
                w_edge = prediction[b, :n_q, 0][edges]
                W_ext = torch.sum(edge_length / 6.0 * (
                    2.0 * q_edge[:, 0] * w_edge[:, 0]
                    + q_edge[:, 0] * w_edge[:, 1]
                    + q_edge[:, 1] * w_edge[:, 0]
                    + 2.0 * q_edge[:, 1] * w_edge[:, 1]
                ))

            Pi = U_ben - W_ext
            energy_terms.append(Pi)
            values = {
                "Pi": Pi,
                "U_int": U_ben,
                "U_mem": torch.zeros_like(U_ben),
                "U_ben": U_ben,
                "U_shr": torch.zeros_like(U_ben),
                "constraint_l2": torch.zeros_like(U_ben),
                "W_ext": W_ext,
            }
            for key, value in values.items():
                stats[key].append(value.detach())

        loss_energy = torch.stack(energy_terms).mean()
        stats_mean = {key: float(torch.stack(value).mean().cpu().item()) for key, value in stats.items()}
        return loss_energy, prediction, stats_mean
