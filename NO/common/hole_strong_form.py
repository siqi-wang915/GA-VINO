"""Sampled strong-form Mindlin--Reissner residuals for ``hole_um_up``.

The mesh is used in full only to identify the candidate interior nodes and
free-hole boundary nodes. The neural operator and coordinate automatic
differentiation see only a fresh collocation set: interior nodes plus sampled
unique hole-boundary nodes. Thus ``max_interior_points`` limits the high-order
AD graph itself.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch

from NO.common.core import MRPlateEnergyOperator, device, plate_isotropic_lame_plane_stress


def _boundary_edges(faces: torch.Tensor) -> torch.Tensor:
    """Return mesh edges owned by exactly one triangle."""
    edges = torch.cat((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), dim=0)
    edges = torch.sort(edges, dim=1).values
    unique, counts = torch.unique(edges, dim=0, return_counts=True)
    return unique[counts == 1]


def _hole_boundary_data(
    verts_xy: torch.Tensor, faces: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return free-hole edges and their outward unit normals."""
    edges = _boundary_edges(faces)
    edge_xy = verts_xy[edges]
    mins = verts_xy.amin(dim=0)
    maxs = verts_xy.amax(dim=0)
    tol = 1.0e-6 * torch.max(maxs - mins).clamp_min(1.0)
    endpoint_on_outer = (
        torch.isclose(edge_xy[..., 0], mins[0], atol=float(tol))
        | torch.isclose(edge_xy[..., 0], maxs[0], atol=float(tol))
        | torch.isclose(edge_xy[..., 1], mins[1], atol=float(tol))
        | torch.isclose(edge_xy[..., 1], maxs[1], atol=float(tol))
    )
    hole_edges = edges[~endpoint_on_outer.all(dim=1)]
    if hole_edges.numel() == 0:
        raise ValueError("hole_um_up strong-form loss could not find a free hole boundary.")

    # Rotate the edge tangent by 90 degrees to obtain an exact normal.  The
    # sole adjacent triangle identifies the material side, so the normal is
    # flipped when necessary to point away from that triangle (out of the
    # material domain and into the hole).
    normals = []
    for i, j in hole_edges.detach().cpu().tolist():
        has_i = (faces == i).any(dim=1)
        has_j = (faces == j).any(dim=1)
        tri = faces[has_i & has_j][0]
        third = tri[(tri != i) & (tri != j)][0]
        midpoint = 0.5 * (verts_xy[i] + verts_xy[j])
        tangent = verts_xy[j] - verts_xy[i]
        outward = torch.stack((tangent[1], -tangent[0]))
        if torch.dot(outward, verts_xy[third] - midpoint) > 0:
            outward = -outward
        normals.append(outward / torch.linalg.vector_norm(outward).clamp_min(1.0e-12))
    return hole_edges, torch.stack(normals, dim=0)


def _hole_boundary_node_data(
    verts_xy: torch.Tensor,
    hole_edges: torch.Tensor,
    edge_normals: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one tangent, outward normal, and arc-length weight per hole node.

    The mesh represents a smooth free boundary by straight segments.  A shared
    vertex must not receive the two distinct normals of its incident segments:
    doing so imposes two pointwise natural conditions at the same network
    query.  Instead, this routine follows each closed boundary loop and uses
    the centred, arc-length-aware tangent at every *unique* boundary node.
    The existing segment normals are used only to orient the resulting node
    normal away from the material and into the hole.
    """
    edge_list = [tuple(map(int, edge)) for edge in hole_edges.detach().cpu().tolist()]
    if not edge_list:
        raise ValueError("No free-hole edges were supplied for node-normal construction.")

    neighbours: Dict[int, List[int]] = {}
    incident_edges: Dict[int, List[int]] = {}
    for edge_idx, (i, j) in enumerate(edge_list):
        neighbours.setdefault(i, []).append(j)
        neighbours.setdefault(j, []).append(i)
        incident_edges.setdefault(i, []).append(edge_idx)
        incident_edges.setdefault(j, []).append(edge_idx)
    malformed = [node for node, adjacent in neighbours.items() if len(adjacent) != 2]
    if malformed:
        raise ValueError(
            "Each free-hole boundary node must have exactly two incident hole edges; "
            f"found invalid nodes {malformed[:5]}."
        )

    loops: List[List[int]] = []
    unseen = set(neighbours)
    while unseen:
        start = next(iter(unseen))
        loop, previous, current = [], None, start
        while True:
            loop.append(current)
            unseen.discard(current)
            adjacent = neighbours[current]
            following = adjacent[0] if adjacent[0] != previous else adjacent[1]
            if following == start:
                break
            if following in loop:
                raise ValueError("Free-hole boundary edges do not form disjoint simple loops.")
            previous, current = current, following
        loops.append(loop)

    node_ids: List[int] = []
    node_normals: List[torch.Tensor] = []
    node_weights: List[torch.Tensor] = []
    for loop in loops:
        n_loop = len(loop)
        if n_loop < 3:
            raise ValueError("A free-hole boundary loop must contain at least three nodes.")
        for i, node in enumerate(loop):
            previous = loop[(i - 1) % n_loop]
            following = loop[(i + 1) % n_loop]
            x_prev, x_node, x_next = verts_xy[previous], verts_xy[node], verts_xy[following]
            l_prev = torch.linalg.vector_norm(x_node - x_prev).clamp_min(1.0e-12)
            l_next = torch.linalg.vector_norm(x_next - x_node).clamp_min(1.0e-12)
            tangent = (x_node - x_prev) / l_prev + (x_next - x_node) / l_next
            if float(torch.linalg.vector_norm(tangent)) <= 1.0e-12:
                tangent = (x_next - x_prev).clone()
            tangent = tangent / torch.linalg.vector_norm(tangent).clamp_min(1.0e-12)
            normal = torch.stack((tangent[1], -tangent[0]))

            # Segment normals were already oriented by their material-side
            # triangle.  Their local average provides a robust sign reference
            # for the centred-difference normal even for a random hole shape.
            reference = edge_normals[incident_edges[node]].mean(dim=0)
            if torch.dot(normal, reference) < 0:
                normal = -normal
            node_ids.append(node)
            node_normals.append(normal)
            node_weights.append(0.5 * (l_prev + l_next))

    return (
        torch.tensor(node_ids, dtype=torch.long, device=hole_edges.device).contiguous(),
        torch.stack(node_normals, dim=0).contiguous(),
        torch.stack(node_weights, dim=0).contiguous(),
    )


def _cpu_tensor(value: Any, *, dtype: torch.dtype) -> torch.Tensor:
    """Return an immutable collocation-cache tensor kept in host memory."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=dtype).contiguous()
    return torch.as_tensor(value, dtype=dtype, device="cpu").contiguous()


def build_hole_strong_collocation_cache(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Precompute mesh-topology candidates for one fixed plate geometry.

    The returned tensors deliberately stay on CPU.  Only the small random
    collocation subset is copied into the model batch later.  The cache is
    independent of the requested sample limits, so interior nodes and
    unique hole-boundary nodes can still be randomly redrawn on every
    optimization step.
    """
    query = sample.get("query", sample.get("pc"))
    if query is None:
        raise KeyError("Strong-form collocation requires sample['query'] or sample['pc'].")

    n_full = int(query.shape[0])
    verts = _cpu_tensor(sample["verts_xy"], dtype=torch.float32)
    faces = _cpu_tensor(sample["faces"], dtype=torch.long)

    boundary_nodes = _boundary_edges(faces).reshape(-1).unique()
    boundary_mask = torch.zeros(n_full, dtype=torch.bool)
    boundary_mask[boundary_nodes] = True
    interior_ids = torch.nonzero(~boundary_mask, as_tuple=False).reshape(-1)
    if interior_ids.numel() == 0:
        raise ValueError("No interior collocation nodes were found.")

    hole_edges, edge_normals = _hole_boundary_data(verts, faces)
    hole_node_ids, hole_node_normals, hole_node_weights = _hole_boundary_node_data(
        verts, hole_edges, edge_normals
    )
    return {
        "interior_ids": interior_ids.contiguous(),
        "hole_node_ids": hole_node_ids,
        "hole_node_normals": hole_node_normals,
        "hole_node_weights": hole_node_weights,
        "n_full": n_full,
    }


def _draw_collocation_plan(
    batch: Dict[str, Any],
    *,
    padding_value: float,
    max_interior_points: int,
    max_hole_edges: int,
) -> List[Dict[str, Any]]:
    """Sample original node IDs before neural evaluation or AD.

    A hole boundary vertex occurs once only.  Its normal is reconstructed from
    its adjacent boundary nodes in ``build_hole_strong_collocation_cache``.
    ``max_hole_edges`` is retained as a backward-compatible configuration name
    but now limits the number of unique hole-boundary nodes.
    """
    query = batch["query"]
    cached_candidates = batch.get("strong_collocation_cache")
    plans: List[Dict[str, Any]] = []
    for b in range(query.shape[0]):
        cached = None if cached_candidates is None else cached_candidates[b]
        if cached is None:
            # Compatibility path for callers that do not use the cached
            # strong-form dataset.  The stage-2 physics trainer always uses
            # the cache below.
            n_full = int((query[b, :, 0] != float(padding_value)).sum().item())
            verts = batch["verts_xy"][b]
            faces = batch["faces"][b].to(dtype=torch.long)
            boundary_nodes = _boundary_edges(faces).reshape(-1).unique()
            boundary_mask = torch.zeros(n_full, dtype=torch.bool, device=faces.device)
            boundary_mask[boundary_nodes] = True
            interior_ids = torch.nonzero(~boundary_mask, as_tuple=False).reshape(-1)
            if interior_ids.numel() == 0:
                raise ValueError("No interior collocation nodes were found.")
            hole_edges, edge_normals = _hole_boundary_data(verts, faces)
            hole_node_ids, hole_node_normals, hole_node_weights = _hole_boundary_node_data(
                verts, hole_edges, edge_normals
            )
        else:
            n_full = int(cached["n_full"])
            interior_ids = cached["interior_ids"]
            hole_node_ids = cached["hole_node_ids"]
            hole_node_normals = cached["hole_node_normals"]
            hole_node_weights = cached["hole_node_weights"]

        if max_interior_points > 0 and interior_ids.numel() > max_interior_points:
            keep = torch.randperm(interior_ids.numel(), device=interior_ids.device)[:max_interior_points]
            interior_ids = interior_ids[keep]
        if max_hole_edges > 0 and hole_node_ids.numel() > max_hole_edges:
            keep = torch.randperm(hole_node_ids.numel(), device=hole_node_ids.device)[:max_hole_edges]
            hole_node_ids = hole_node_ids[keep]
            hole_node_normals = hole_node_normals[keep]
            hole_node_weights = hole_node_weights[keep]

        plans.append(
            {
                "query_ids": torch.cat((interior_ids, hole_node_ids), dim=0),
                "n_interior": int(interior_ids.numel()),
                "n_hole_nodes": int(hole_node_ids.numel()),
                "hole_node_normals": hole_node_normals,
                "hole_node_weights": hole_node_weights,
                "n_full": n_full,
            }
        )
    return plans


def _slice_query_aligned_fields(
    batch: Dict[str, Any],
    plans: List[Dict[str, Any]],
    *,
    padding_value: float,
) -> Dict[str, Any]:
    """Build a collocation batch while retaining full geometry/operator inputs."""
    sampled = dict(batch)
    max_points = max(int(plan["query_ids"].numel()) for plan in plans)
    B = len(plans)
    # pc_net/load_pc_net/material_pc_net are operator inputs, not output-query
    # fields, and are deliberately not downsampled here.
    for key in ("query", "query_net", "bc_factor", "pressure", "target_fem_local"):
        if key not in batch:
            continue
        source = batch[key]
        out = torch.full(
            (B, max_points, *source.shape[2:]),
            0.0 if key == "bc_factor" else float(padding_value),
            dtype=source.dtype,
            device=source.device,
        )
        for b, plan in enumerate(plans):
            ids = plan["query_ids"].to(device=source.device)
            out[b, : ids.numel()] = source[b, ids]
        sampled[key] = out

    interior_mask = torch.zeros((B, max_points), dtype=torch.bool, device=batch["query"].device)
    for b, plan in enumerate(plans):
        interior_mask[b, : plan["n_interior"]] = True
    sampled["strong_interior_mask"] = interior_mask
    return sampled


def _grad_selected(values: torch.Tensor, query_net: torch.Tensor, counts: List[int]) -> torch.Tensor:
    """Differentiate only valid prefixes of batched collocation tensors."""
    total = torch.cat([values[b, :count].reshape(-1) for b, count in enumerate(counts)]).sum()
    return torch.autograd.grad(total, query_net, create_graph=True, retain_graph=True)[0]


def hole_um_up_strong_form_loss(
    operator_model: MRPlateEnergyOperator,
    batch: Dict,
    *,
    padding_value: float,
    max_interior_points: int = 512,
    max_hole_edges: int = 128,
    interior_weight: float = 1.0,
    rotation_x_weight: float = 1.0,
    rotation_y_weight: float = 1.0,
    free_shear_weight: float = 1.0,
    free_moment_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, Dict[str, Any]]:
    """Return sampled strong loss, sampled prediction, and collocation batch.

    ``strong_data`` uses the returned batch so its FEM targets are indexed by
    the same sampled node IDs. Its data MSE is limited to random interior
    points; selected hole-boundary nodes are reserved for natural-boundary
    loss.
    """
    plans = _draw_collocation_plan(
        batch,
        padding_value=padding_value,
        max_interior_points=max_interior_points,
        max_hole_edges=max_hole_edges,
    )
    grad_batch = _slice_query_aligned_fields(batch, plans, padding_value=padding_value)
    query_net = grad_batch["query_net"].to(device).detach().clone().requires_grad_(True)
    grad_batch["query_net"] = query_net

    raw = operator_model.model(grad_batch)
    pred = operator_model.apply_boundary_transform(raw, grad_batch)
    all_counts = [int(plan["n_interior"] + plan["n_hole_nodes"]) for plan in plans]
    interior_counts = [int(plan["n_interior"]) for plan in plans]

    # First derivatives are needed at interior and unique hole-boundary nodes.
    dw = _grad_selected(pred[..., 0], query_net, all_counts)
    dthx = _grad_selected(pred[..., 1], query_net, all_counts)
    dthy = _grad_selected(pred[..., 2], query_net, all_counts)

    B = pred.shape[0]
    sx, sy, lam, mu, shear, bend = [], [], [], [], [], []
    for b in range(B):
        bbox = batch["bbox"][b].to(device=device, dtype=pred.dtype)
        sx.append(2.0 / (bbox[1] - bbox[0]).clamp_min(1.0e-12))
        sy.append(2.0 / (bbox[3] - bbox[2]).clamp_min(1.0e-12))
        E = batch["E"][b, 0].to(device=device, dtype=pred.dtype)
        nu = batch["nu"][b, 0].to(device=device, dtype=pred.dtype)
        h = batch["thickness"][b, 0].to(device=device, dtype=pred.dtype)
        lam_b, mu_b = plate_isotropic_lame_plane_stress(E, nu)
        lam.append(lam_b)
        mu.append(mu_b)
        shear.append((5.0 / 6.0) * mu_b * h)
        bend.append(h**3 / 12.0)
    sx_t, sy_t = torch.stack(sx).view(B, 1), torch.stack(sy).view(B, 1)
    lam_t, mu_t = torch.stack(lam).view(B, 1), torch.stack(mu).view(B, 1)
    shear_t, bend_t = torch.stack(shear).view(B, 1), torch.stack(bend).view(B, 1)

    wx, wy = dw[..., 0] * sx_t, dw[..., 1] * sy_t
    thx_x, thx_y = dthx[..., 0] * sx_t, dthx[..., 1] * sy_t
    thy_x, thy_y = dthy[..., 0] * sx_t, dthy[..., 1] * sy_t
    qx = shear_t * (wx - pred[..., 1])
    qy = shear_t * (wy - pred[..., 2])
    trace_k = thx_x + thy_y
    mxx = bend_t * (lam_t * trace_k + 2.0 * mu_t * thx_x)
    myy = bend_t * (lam_t * trace_k + 2.0 * mu_t * thy_y)
    mxy = bend_t * mu_t * (thx_y + thy_x)

    # Only interior points need a second coordinate derivative.
    dqx = _grad_selected(qx, query_net, interior_counts)
    dqy = _grad_selected(qy, query_net, interior_counts)
    dmxx = _grad_selected(mxx, query_net, interior_counts)
    dmyy = _grad_selected(myy, query_net, interior_counts)
    dmxy = _grad_selected(mxy, query_net, interior_counts)

    total_interior = pred.new_zeros(())
    total_rotation_x = pred.new_zeros(())
    total_rotation_y = pred.new_zeros(())
    total_shear_bc = pred.new_zeros(())
    total_moment_bc = pred.new_zeros(())
    for b, plan in enumerate(plans):
        n_interior = int(plan["n_interior"])
        n_hole_nodes = int(plan["n_hole_nodes"])
        sx_b, sy_b = sx_t[b, 0], sy_t[b, 0]
        div_q = dqx[b, :n_interior, 0] * sx_b + dqy[b, :n_interior, 1] * sy_b
        div_mx = dmxx[b, :n_interior, 0] * sx_b + dmxy[b, :n_interior, 1] * sy_b
        div_my = dmxy[b, :n_interior, 0] * sx_b + dmyy[b, :n_interior, 1] * sy_b

        pressure_full = batch["pressure"][b, : plan["n_full"], 0].to(device=device, dtype=pred.dtype)
        pressure_interior = grad_batch["pressure"][b, :n_interior, 0].to(device=device, dtype=pred.dtype)
        p_ref = pressure_full.abs().amax().clamp_min(1.0e-12)
        bbox = batch["bbox"][b].to(device=device, dtype=pred.dtype)
        length_ref = torch.sqrt(((bbox[1] - bbox[0]) * (bbox[3] - bbox[2])).clamp_min(1.0e-12))

        rw = div_q + pressure_interior
        rthx = div_mx + qx[b, :n_interior]
        rthy = div_my + qy[b, :n_interior]
        total_interior = total_interior + torch.mean((rw / p_ref) ** 2)
        total_rotation_x = total_rotation_x + torch.mean((rthx / (p_ref * length_ref)) ** 2)
        total_rotation_y = total_rotation_y + torch.mean((rthy / (p_ref * length_ref)) ** 2)

        boundary_start = n_interior
        boundary_stop = boundary_start + n_hole_nodes
        normals = plan["hole_node_normals"].to(device=device, dtype=pred.dtype)
        node_weights = plan["hole_node_weights"].to(device=device, dtype=pred.dtype)
        weight_sum = node_weights.sum().clamp_min(1.0e-12)
        qx_node = qx[b, boundary_start:boundary_stop]
        qy_node = qy[b, boundary_start:boundary_stop]
        mxx_node = mxx[b, boundary_start:boundary_stop]
        myy_node = myy[b, boundary_start:boundary_stop]
        mxy_node = mxy[b, boundary_start:boundary_stop]
        qn = qx_node * normals[:, 0] + qy_node * normals[:, 1]
        mnx = mxx_node * normals[:, 0] + mxy_node * normals[:, 1]
        mny = mxy_node * normals[:, 0] + myy_node * normals[:, 1]
        total_shear_bc = total_shear_bc + torch.sum(
            node_weights * qn.square() / (p_ref * length_ref) ** 2
        ) / weight_sum
        # The factor 1/2 preserves the former per-component mean convention
        # while the node weights now approximate the boundary integral.
        total_moment_bc = total_moment_bc + torch.sum(
            node_weights * 0.5 * (mnx.square() + mny.square())
            / (p_ref * length_ref**2) ** 2
        ) / weight_sum

    denom = float(B)
    interior = total_interior / denom
    rotation_x = total_rotation_x / denom
    rotation_y = total_rotation_y / denom
    shear_bc = total_shear_bc / denom
    moment_bc = total_moment_bc / denom
    weighted_interior = float(interior_weight) * interior
    weighted_rotation_x = float(rotation_x_weight) * rotation_x
    weighted_rotation_y = float(rotation_y_weight) * rotation_y
    weighted_shear_bc = float(free_shear_weight) * shear_bc
    weighted_moment_bc = float(free_moment_weight) * moment_bc
    strong = (
        weighted_interior
        + weighted_rotation_x
        + weighted_rotation_y
        + weighted_shear_bc
        + weighted_moment_bc
    )
    return strong, {
        "strong_interior": interior,
        "strong_rotation_x": rotation_x,
        "strong_rotation_y": rotation_y,
        "strong_free_shear": shear_bc,
        "strong_free_moment": moment_bc,
        "strong_interior_weighted": weighted_interior,
        "strong_rotation_x_weighted": weighted_rotation_x,
        "strong_rotation_y_weighted": weighted_rotation_y,
        "strong_free_shear_weighted": weighted_shear_bc,
        "strong_free_moment_weighted": weighted_moment_bc,
    }, pred, grad_batch
