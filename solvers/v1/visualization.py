"""
visualization.py
Simple visualization of the elastica solution space results.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_results(npz_file="solution_space_results.npz", d_slice_indices=None,
                 d_slice_values=None, output_file="solution_space_visualization.png"):
    """
    Load results and create plots for each scalar field.

    Parameters
    ----------
    npz_file : str
        Path to the results .npz file.
    d_slice_indices : int, list[int], or None
        Index or indices of the d-slice(s) to plot. If None, the middle d
        slice is used.
    d_slice_values : float, list[float], or None
        d value or values to plot. Each value is mapped to the nearest value
        in the stored d grid. Cannot be used together with d_slice_indices.
    output_file : str
        Path for the generated visualization image.
    """
    data = np.load(npz_file)
    coords = data["coords"]          # (N, 3) array of (d, phi1, phi2) for each node
    ijk = data["ijk"]                # (N, 3) array of grid indices for each node
    grid_shape = data["grid_shape"]  # (nd, n1, n2)
    solved = data["solved"]          # (N,) boolean array
    fx = data["fx"]                  # (N,) array
    fy = data["fy"]                  # (N,) array
    moment = data["moment"]          # (N,) array
    energy = data["energy"]          # (N,) array
    d_vals = data["d_vals"]
    p1_vals = data["p1_vals"]
    p2_vals = data["p2_vals"]

    nd, n1, n2 = grid_shape

    if d_slice_indices is not None and d_slice_values is not None:
        raise ValueError("Specify d_slice_indices or d_slice_values, not both")

    if d_slice_values is not None:
        if np.isscalar(d_slice_values):
            d_slice_values = [float(d_slice_values)]
        d_slice_indices = [int(np.abs(d_vals - value).argmin())
                           for value in d_slice_values]

    if d_slice_indices is None:
        d_slice_indices = [nd // 2]
    elif np.isscalar(d_slice_indices):
        d_slice_indices = [int(d_slice_indices)]
    else:
        d_slice_indices = [int(idx) for idx in d_slice_indices]

    d_slice_indices = [max(0, min(int(idx), nd - 1)) for idx in d_slice_indices]
    d_slice_indices = list(dict.fromkeys(d_slice_indices))

    # We'll create a figure with subplots for each scalar field
    fields = [
        (fx, "fx", "Horizontal force multiplier"),
        (fy, "fy", "Vertical force multiplier"),
        (moment, "moment", "Bending moment at s=0"),
        (energy, "energy", "Bending strain energy")
    ]

    n_fields = len(fields)
    n_slice_plots = len(d_slice_indices)
    n_cols = 1 + n_slice_plots
    fig = plt.figure(figsize=(6 * n_cols, 5 * n_fields), constrained_layout=True)

    for idx, (field, label, title) in enumerate(fields):
        ax_3d = fig.add_subplot(n_fields, n_cols, idx * n_cols + 1, projection='3d')
        solved_coords = coords[solved]
        solved_field = field[solved]
        if len(solved_coords) > 0:
            finite = np.isfinite(solved_field)
            if np.any(finite):
                vmin = np.nanmin(solved_field[finite])
                vmax = np.nanmax(solved_field[finite])
                if np.isclose(vmin, vmax):
                    vmin, vmax = vmin - 1, vmax + 1
                sc_3d = ax_3d.scatter(solved_coords[finite, 1], solved_coords[finite, 2], solved_coords[finite, 0],
                                      c=solved_field[finite], cmap='viridis', s=12, vmin=vmin, vmax=vmax)
                ax_3d.set_zlabel('d')
                ax_3d.set_xlabel('phi1')
                ax_3d.set_ylabel('phi2')
                ax_3d.set_title(f'{title} (3D solved points)')
                plt.colorbar(sc_3d, ax=ax_3d, label=label)
            else:
                ax_3d.text(0.5, 0.5, 0.5, 'No finite values', transform=ax_3d.transAxes, ha='center')
                ax_3d.set_title(f'{title} (3D solved points)')
        else:
            ax_3d.text(0.5, 0.5, 0.5, 'No solved points', transform=ax_3d.transAxes, ha='center')
            ax_3d.set_title(f'{title} (3D solved points)')

        for j, d_idx in enumerate(d_slice_indices):
            slice_mask = (ijk[:, 0] == d_idx) & solved
            slice_coords = coords[slice_mask]
            slice_field = field[slice_mask]
            ax_slice = fig.add_subplot(n_fields, n_cols, idx * n_cols + 2 + j)
            if len(slice_coords) > 0:
                finite = np.isfinite(slice_field)
                if np.any(finite):
                    vmin = np.nanmin(slice_field[finite])
                    vmax = np.nanmax(slice_field[finite])
                    if np.isclose(vmin, vmax):
                        vmin, vmax = vmin - 1, vmax + 1
                    sc_slice = ax_slice.scatter(slice_coords[finite, 1], slice_coords[finite, 2],
                                               c=slice_field[finite], cmap='viridis', s=12,
                                               vmin=vmin, vmax=vmax)
                    ax_slice.set_xlabel('phi1')
                    ax_slice.set_ylabel('phi2')
                    ax_slice.set_title(f'{title} (d = {d_vals[d_idx]:.3f})')
                    plt.colorbar(sc_slice, ax=ax_slice, label=label)
                else:
                    ax_slice.text(0.5, 0.5, 'No finite values', transform=ax_slice.transAxes, ha='center')
                    ax_slice.set_title(f'{title} (d = {d_vals[d_idx]:.3f})')
            else:
                ax_slice.text(0.5, 0.5, 'No solved points in slice', transform=ax_slice.transAxes, ha='center')
                ax_slice.set_title(f'{title} (d = {d_vals[d_idx]:.3f})')

    plt.savefig(output_file, dpi=200)
    plt.close()

    print(f"Visualization saved to {output_file}")


if __name__ == "__main__":
    plot_results()