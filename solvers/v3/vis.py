"""All matplotlib visualizations for solution-space results."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.widgets import Button, RadioButtons, Slider, TextBox


FIELD_INFO = {
    "fx": ("fx", "Horizontal force multiplier"),
    "fy": ("fy", "Vertical force multiplier"),
    "m": ("moment", "Bending moment at s=0"),
    "U": ("energy", "Bending strain energy"),
    "theta": ("theta_gradient_score", "RMS d theta / d(phi1,phi2)"),
}


def plot_results(npz_file="solution_space_results.npz", d_slice_indices=None,
                 d_slice_values=None,
                 output_file="solution_space_visualization.png"):
    """Create scalar-field plots for the solved parameter space."""
    data = np.load(npz_file)
    required = {"coords", "ijk", "grid_shape", "solved", "fx", "fy",
                "moment", "energy", "d_vals", "p1_vals", "p2_vals"}
    missing = required.difference(data.files)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(
            f"{npz_file} is a mesh archive, not a solved-results archive; "
            f"missing: {missing_text}. Run `python main.py "
            f"config_adaptive.json` first, then plot "
            "solution_space_results_ar.npz."
        )
    coords = data["coords"]
    ijk = data["ijk"]
    grid_shape = data["grid_shape"]
    solved = data["solved"]
    fields = [
        (data["fx"], "fx", "Horizontal force multiplier"),
        (data["fy"], "fy", "Vertical force multiplier"),
        (data["moment"], "moment", "Bending moment at s=0"),
        (data["energy"], "energy", "Bending strain energy"),
    ]
    if "refinement_count" in data.files:
        fields.append(
            (data["refinement_count"], "refinements", "Adaptive refinements")
        )
    d_vals = data["d_vals"]
    p1_vals = data["p1_vals"]
    p2_vals = data["p2_vals"]
    nd, _, _ = grid_shape
    refinement_params, refinement_fields, refinement_slices = (
        _load_refinement_plot_data(data, d_vals)
    )

    if d_slice_indices is not None and d_slice_values is not None:
        raise ValueError("Specify d_slice_indices or d_slice_values, not both")
    if d_slice_values is not None:
        if np.isscalar(d_slice_values):
            d_slice_values = [float(d_slice_values)]
        d_slice_indices = [int(np.abs(d_vals - value).argmin())
                           for value in d_slice_values]
    if d_slice_indices is None:
        d_slice_indices = list(range(nd))
    elif np.isscalar(d_slice_indices):
        d_slice_indices = [int(d_slice_indices)]
    else:
        d_slice_indices = [int(index) for index in d_slice_indices]
    d_slice_indices = [max(0, min(index, nd - 1)) for index in d_slice_indices]
    d_slice_indices = list(dict.fromkeys(d_slice_indices))

    field_norms = {}
    for field, label, _ in fields:
        finite = field[solved & np.isfinite(field)]
        if finite.size == 0:
            field_norms[label] = None
            continue
        vmin, vmax = float(np.min(finite)), float(np.max(finite))
        if np.isclose(vmin, vmax):
            vmin -= 1.0
            vmax += 1.0
        field_norms[label] = Normalize(vmin=vmin, vmax=vmax)

    n_fields = len(fields)
    n_slice_plots = len(d_slice_indices)
    n_cols = 2 + n_slice_plots
    fig = plt.figure(figsize=(6.5 * n_cols, 5 * n_fields),
                     constrained_layout=True)
    grid = fig.add_gridspec(
        n_fields, n_cols, width_ratios=[1.0, 0.18] + [1.0] * n_slice_plots,
        wspace=0.35,
    )

    for row, (field, label, title) in enumerate(fields):
        ax_3d = fig.add_subplot(grid[row, 0], projection="3d")
        row_axes = [ax_3d]
        ax_3d.set_xlim(float(p1_vals.min()), float(p1_vals.max()))
        ax_3d.set_ylim(float(p2_vals.min()), float(p2_vals.max()))
        ax_3d.set_zlim(float(d_vals.min()), float(d_vals.max()))
        solved_coords, solved_field = coords[solved], field[solved]
        norm = field_norms[label]
        finite = np.isfinite(solved_field)
        if finite.any():
            ax_3d.scatter(
                solved_coords[finite, 1], solved_coords[finite, 2],
                solved_coords[finite, 0], c=solved_field[finite],
                cmap="viridis", s=12, norm=norm,
            )
            refinement_field = refinement_fields.get(label)
            if refinement_field is not None:
                refinement_finite = np.isfinite(refinement_field)
                if refinement_finite.any():
                    ax_3d.scatter(
                        refinement_params[refinement_finite, 1],
                        refinement_params[refinement_finite, 2],
                        refinement_params[refinement_finite, 0],
                        c=refinement_field[refinement_finite],
                        cmap="viridis", norm=norm, marker="x", s=24,
                        linewidths=0.8,
                    )
            ax_3d.set_zlabel("d")
            ax_3d.set_xlabel("phi1")
            ax_3d.set_ylabel("phi2")
            ax_3d.set_title(f"{title} (3D solved points)")
        else:
            ax_3d.text(0.5, 0.5, 0.5, "No finite values",
                       transform=ax_3d.transAxes, ha="center")
            ax_3d.set_title(f"{title} (3D solved points)")

        for column, d_idx in enumerate(d_slice_indices):
            slice_mask = (ijk[:, 0] == d_idx) & solved
            slice_coords, slice_field = coords[slice_mask], field[slice_mask]
            ax_slice = fig.add_subplot(grid[row, column + 2])
            row_axes.append(ax_slice)
            ax_slice.set_xlim(float(p1_vals.min()), float(p1_vals.max()))
            ax_slice.set_ylim(float(p2_vals.min()), float(p2_vals.max()))
            finite = np.isfinite(slice_field)
            if finite.any():
                ax_slice.scatter(
                    slice_coords[finite, 1], slice_coords[finite, 2],
                    c=slice_field[finite], cmap="viridis", s=12, norm=norm,
                )
                refinement_field = refinement_fields.get(label)
                if refinement_field is not None:
                    refinement_mask = refinement_slices == d_idx
                    refinement_finite = refinement_mask & np.isfinite(refinement_field)
                    if refinement_finite.any():
                        ax_slice.scatter(
                            refinement_params[refinement_finite, 1],
                            refinement_params[refinement_finite, 2],
                            c=refinement_field[refinement_finite],
                            cmap="viridis", norm=norm, marker="x", s=28,
                            linewidths=0.8,
                        )
                ax_slice.set_xlabel("phi1")
                ax_slice.set_ylabel("phi2")
                ax_slice.set_title(f"{title} (d = {d_vals[d_idx]:.3f})")
            else:
                ax_slice.text(0.5, 0.5, "No finite values",
                              transform=ax_slice.transAxes, ha="center")
                ax_slice.set_title(f"{title} (d = {d_vals[d_idx]:.3f})")

        if norm is not None:
            mappable = ScalarMappable(norm=norm, cmap="viridis")
            mappable.set_array([])
            fig.colorbar(mappable, ax=row_axes, label=label)

    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Visualization saved to {output_file}")


def _load_refinement_plot_data(data, d_vals):
    """Return refinement positions, scalar values, and nearest d-slices."""
    required = {"refinement_params", "refinement_steps", "refinement_theta",
                "refinement_fx", "refinement_fy"}
    if not required.issubset(data.files):
        return np.empty((0, 3)), {}, np.empty(0, dtype=int)

    refinement_params = np.asarray(data["refinement_params"], dtype=float)
    refinement_theta = np.asarray(data["refinement_theta"], dtype=float)
    refinement_steps = np.asarray(data["refinement_steps"], dtype=float)
    refinement_fields = {
        "fx": np.asarray(data["refinement_fx"], dtype=float),
        "fy": np.asarray(data["refinement_fy"], dtype=float),
        "refinements": refinement_steps + 1.0,
    }
    if refinement_theta.ndim == 2 and refinement_theta.shape[0]:
        beam_x = np.asarray(data["beam_x"], dtype=float)
        spacing = np.diff(beam_x)
        slopes = np.diff(refinement_theta, axis=1) / spacing[None, :]
        refinement_fields["moment"] = slopes[:, 0]
        refinement_fields["energy"] = 0.5 * np.sum(
            0.5 * (slopes[:, :-1] ** 2 + slopes[:, 1:] ** 2)
            * spacing[None, 1:], axis=1,
        )
    else:
        for label in ("moment", "energy"):
            refinement_fields[label] = np.full(refinement_params.shape[0], np.nan)

    if d_vals is None or refinement_params.size == 0:
        refinement_slices = np.empty(0, dtype=int)
    else:
        refinement_slices = np.abs(
            np.asarray(d_vals)[:, None] - refinement_params[:, 0]
        ).argmin(axis=0)
    return refinement_params, refinement_fields, refinement_slices


def load_gradient_results(path):
    data = np.load(path)
    required = {"ijk", "grid_shape", "d_vals", "p1_vals", "p2_vals",
                "solved", *(name for name, _ in FIELD_INFO.values())}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"{path} is missing required result arrays: "
                         f"{', '.join(sorted(missing))}")
    return data


def gradient_grid(data, field_name, d_index):
    """Return values and phi1/phi2 gradients for one d slice."""
    nd, n1, n2 = (int(value) for value in data["grid_shape"])
    if not 0 <= d_index < nd:
        raise IndexError(f"d index {d_index} is outside [0, {nd - 1}]")
    values = np.full((n1, n2), np.nan, dtype=float)
    ijk, solved = data["ijk"], data["solved"]
    node_indices = np.flatnonzero((ijk[:, 0] == d_index) & solved)
    values[ijk[node_indices, 1], ijk[node_indices, 2]] = \
        data[field_name][node_indices]
    gradient_phi1, gradient_phi2 = np.gradient(
        values, data["p1_vals"], data["p2_vals"], axis=(0, 1), edge_order=1
    )
    return values, gradient_phi1, gradient_phi2


def run_gradient_viewer(npz_file="solution_space_results.npz"):
    """Show the interactive gradient heatmap and contour viewer."""
    data = load_gradient_results(npz_file)
    d_vals = np.asarray(data["d_vals"], dtype=float)
    selected_field = "fx"
    fig, (ax, contour_ax) = plt.subplots(1, 2, figsize=(13, 7))
    fig.subplots_adjust(left=0.08, right=0.79, bottom=0.23, wspace=0.28)
    slider_ax = fig.add_axes((0.16, 0.11, 0.56, 0.035))
    d_slider = Slider(slider_ax, "d slice", 0, len(d_vals) - 1,
                      valinit=0, valstep=1, valfmt="%0.0f")
    radio_ax = fig.add_axes((0.84, 0.38, 0.13, 0.25))
    field_radio = RadioButtons(radio_ax, list(FIELD_INFO), active=0)
    radio_ax.set_title("field")
    colorbar_ax = fig.add_axes((0.445, 0.25, 0.018, 0.50))
    colorbar_ax.set_visible(False)
    colorbar = None

    def draw(_value=None):
        nonlocal colorbar
        d_index = int(d_slider.val)
        field_key = selected_field
        field_name, field_title = FIELD_INFO[field_key]
        values, grad_phi1, grad_phi2 = gradient_grid(data, field_name, d_index)
        finite = np.isfinite(values) & np.isfinite(grad_phi1) & np.isfinite(grad_phi2)
        ax.clear()
        contour_ax.clear()
        if not finite.any():
            colorbar_ax.set_visible(False)
            ax.set_title(f"{field_key}: no solved nodes at d = {d_vals[d_index]:.6g}")
            contour_ax.set_title("no gradient data")
        else:
            phi1_grid, phi2_grid = np.meshgrid(
                data["p1_vals"], data["p2_vals"], indexing="ij"
            )
            magnitude = np.hypot(grad_phi1, grad_phi2)
            magnitude[~finite] = np.nan
            finite_magnitude = magnitude[np.isfinite(magnitude)]
            norm = Normalize(float(np.min(finite_magnitude)),
                             float(np.max(finite_magnitude)))
            if np.isclose(norm.vmin, norm.vmax):
                norm = Normalize(norm.vmin - 1.0, norm.vmax + 1.0)
            heatmap = ax.pcolormesh(phi1_grid, phi2_grid, magnitude,
                                    cmap="viridis", norm=norm,
                                    shading="gouraud")
            contour_ax.contour(phi1_grid, phi2_grid,
                               np.ma.masked_where(~finite, magnitude),
                               levels=40, cmap="viridis", norm=norm,
                               linewidths=0.75)
            colorbar_ax.set_visible(True)
            if colorbar is None:
                colorbar = fig.colorbar(heatmap, cax=colorbar_ax)
            else:
                colorbar.update_normal(heatmap)
            colorbar.set_label("gradient magnitude")
            ax.set_title(f"grad({field_key}) heatmap | {field_title} | "
                         f"d = {d_vals[d_index]:.6g}")
            contour_ax.set_title(f"grad({field_key}) line contours | "
                                 f"d = {d_vals[d_index]:.6g}")
        for plot_ax in (ax, contour_ax):
            plot_ax.set_xlabel("phi1")
            plot_ax.set_ylabel("phi2")
            plot_ax.set_xlim(float(data["p1_vals"].min()),
                             float(data["p1_vals"].max()))
            plot_ax.set_ylim(float(data["p2_vals"].min()),
                             float(data["p2_vals"].max()))
            plot_ax.set_aspect("equal", adjustable="box")
            plot_ax.grid(alpha=0.25)
        fig.canvas.draw_idle()

    def select_field(label):
        nonlocal selected_field
        selected_field = label
        draw()

    d_slider.on_changed(draw)
    field_radio.on_clicked(select_field)
    draw()
    plt.show()


def load_interactive_results(path):
    data = np.load(path)
    required = {"coords", "ijk", "grid_shape", "d_vals", "solved",
                "theta", "energy"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"{path} does not contain theta data ({', '.join(sorted(missing))}).")
    beam_x = data["beam_x"] if "beam_x" in data else np.linspace(
        0.0, 1.0, data["theta"].shape[1]
    )
    return data, beam_x


def elastica_shape(beam_x, theta):
    dx = np.diff(beam_x)
    shape_x = np.concatenate(([0.0], np.cumsum(
        0.5 * (np.cos(theta[:-1]) + np.cos(theta[1:])) * dx)))
    shape_y = np.concatenate(([0.0], np.cumsum(
        0.5 * (np.sin(theta[:-1]) + np.sin(theta[1:])) * dx)))
    return shape_x, shape_y


def run_interactive_viewer(npz_file="solution_space_results.npz"):
    """Show solved slices and the selected elastica shape."""
    data, beam_x = load_interactive_results(npz_file)
    coords, ijk = data["coords"], data["ijk"]
    d_vals, solved = data["d_vals"], data["solved"]
    theta, energy = data["theta"], data["energy"]
    refinement_nodes = data["refinement_nodes"] if "refinement_nodes" in data.files else np.array([], dtype=int)
    refinement_theta = data["refinement_theta"] if "refinement_theta" in data.files else np.empty((0, theta.shape[1]))
    nd = len(d_vals)
    finite_energy = energy[solved & np.isfinite(energy)]
    if not finite_energy.size:
        raise ValueError("The results file contains no finite bending energies.")
    energy_norm = Normalize(float(np.min(finite_energy)),
                            float(np.max(finite_energy)))
    if np.isclose(energy_norm.vmin, energy_norm.vmax):
        energy_norm = Normalize(energy_norm.vmin - 1.0,
                                energy_norm.vmax + 1.0)

    fig, (slice_ax, shape_ax) = plt.subplots(1, 2, figsize=(13, 6))
    fig.subplots_adjust(bottom=0.25, right=0.93, wspace=0.28)
    d_slider = Slider(fig.add_axes((0.16, 0.06, 0.68, 0.035)),
                      "d slice", 0, nd - 1, valinit=0, valstep=1,
                      valfmt="%0.0f")
    phi1_box = TextBox(fig.add_axes((0.16, 0.13, 0.14, 0.04)),
                       "phi1", initial="0.0")
    phi2_box = TextBox(fig.add_axes((0.34, 0.13, 0.14, 0.04)),
                       "phi2", initial="0.0")
    view_button = Button(fig.add_axes((0.52, 0.13, 0.12, 0.04)), "View")
    colorbar_ax = fig.add_axes((0.47, 0.25, 0.018, 0.50))
    colorbar_ax.set_visible(False)
    selected_artist = None
    points_artist = None
    energy_colorbar = None
    current_slice_indices = np.array([], dtype=int)

    def solved_nodes_in_slice(d_index):
        d_index = max(0, min(int(d_index), nd - 1))
        return np.flatnonzero(solved & np.isclose(coords[:, 0], d_vals[d_index]))

    def set_shape(node_indices, title, refinement_indices=()):
        shape_ax.clear()
        for refinement_index in refinement_indices:
            shape_x, shape_y = elastica_shape(
                beam_x, refinement_theta[int(refinement_index)]
            )
            shape_ax.plot(shape_x, shape_y, color="tab:purple", alpha=0.35,
                          linewidth=1)
        for node_index in np.atleast_1d(node_indices):
            shape_x, shape_y = elastica_shape(beam_x, theta[int(node_index)])
            shape_ax.plot(shape_x, shape_y, color="tab:blue", alpha=0.3,
                          linewidth=1)
        node_index = int(np.atleast_1d(node_indices)[-1])
        shape_x, shape_y = elastica_shape(beam_x, theta[node_index])
        shape_ax.plot(shape_x, shape_y, color="tab:orange", linewidth=2.2)
        shape_ax.scatter([shape_x[0], shape_x[-1]], [shape_y[0], shape_y[-1]],
                         color=["tab:green", "tab:red"], zorder=3)
        shape_ax.set_title(title)
        shape_ax.set_xlabel("x")
        shape_ax.set_ylabel("y")
        shape_ax.set_aspect("equal", adjustable="box")
        shape_ax.set_xlim(-0.2, 1.0)
        shape_ax.set_ylim(-0.3, 0.3)
        shape_ax.grid(alpha=0.25)

    def draw_slice(d_index):
        nonlocal points_artist, energy_colorbar, current_slice_indices
        d_index = int(d_index)
        current_slice_indices = solved_nodes_in_slice(d_index)
        shape_ax.clear()
        if current_slice_indices.size == 0:
            if points_artist is not None:
                points_artist.set_offsets(np.empty((0, 2)))
                points_artist.set_array(np.array([], dtype=float))
            colorbar_ax.set_visible(False)
            slice_ax.set_title(f"d = {d_vals[d_index]:.6g}: no solved nodes")
        else:
            offsets = coords[current_slice_indices][:, 1:3]
            values = energy[current_slice_indices]
            if points_artist is None:
                points_artist = slice_ax.scatter(
                    offsets[:, 0], offsets[:, 1], c=values, cmap="viridis",
                    norm=energy_norm, s=12, picker=True,
                )
                energy_colorbar = fig.colorbar(points_artist, cax=colorbar_ax)
                energy_colorbar.set_label("bending energy")
            else:
                points_artist.set_offsets(offsets)
                points_artist.set_array(values)
            colorbar_ax.set_visible(True)
            slice_ax.set_title(f"d = {d_vals[d_index]:.6g} | click a solved node")
        slice_ax.set_xlabel("phi1")
        slice_ax.set_ylabel("phi2")
        slice_ax.set_xlim(float(coords[:, 1].min()), float(coords[:, 1].max()))
        slice_ax.set_ylim(float(coords[:, 2].min()), float(coords[:, 2].max()))
        slice_ax.set_aspect("equal", adjustable="box")
        slice_ax.grid(alpha=0.25)
        shape_ax.set_title("Select a node")
        shape_ax.set_xlabel("x")
        shape_ax.set_ylabel("y")
        shape_ax.set_aspect("equal", adjustable="box")
        shape_ax.set_xlim(-1.0, 1.0)
        shape_ax.set_ylim(-1.0, 1.0)
        shape_ax.grid(alpha=0.25)
        fig.canvas.draw_idle()

    def show_node(node_index):
        nonlocal selected_artist
        if selected_artist is not None:
            selected_artist.remove()
        selected_artist, = slice_ax.plot(
            coords[node_index, 1], coords[node_index, 2], marker="o",
            markersize=9, markerfacecolor="none", markeredgecolor="red",
            markeredgewidth=2,
        )
        d_value, phi1, phi2 = coords[node_index]
        refinement_indices = np.flatnonzero(refinement_nodes == node_index)
        set_shape([node_index], f"node {node_index}: d={d_value:.6g}, "
                  f"phi1={phi1:.6g}, phi2={phi2:.6g}, "
              f"energy={energy[node_index]:.6g}", refinement_indices)
        fig.canvas.draw_idle()

    def on_click(event):
        if (event.inaxes is not slice_ax or points_artist is None or
                current_slice_indices.size == 0):
            return
        contains, details = points_artist.contains(event)
        if contains and details["ind"].size:
            show_node(int(current_slice_indices[details["ind"][0]]))

    def view_coordinates(_event):
        try:
            phi1_value, phi2_value = float(phi1_box.text), float(phi2_box.text)
        except ValueError:
            shape_ax.set_title("Enter numeric phi1 and phi2 values")
            fig.canvas.draw_idle()
            return
        if current_slice_indices.size == 0:
            shape_ax.set_title("No solved nodes in the selected d slice")
            fig.canvas.draw_idle()
            return
        distances = ((coords[current_slice_indices, 1] - phi1_value) ** 2 +
                     (coords[current_slice_indices, 2] - phi2_value) ** 2)
        show_node(int(current_slice_indices[np.argmin(distances)]))

    d_slider.on_changed(draw_slice)
    view_button.on_clicked(view_coordinates)
    fig.canvas.mpl_connect("button_press_event", on_click)
    draw_slice(0)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plot", "gradient", "interactive"),
                        nargs="?", default="plot")
    parser.add_argument("npz_file", nargs="?", default="solution_space_results.npz")
    args = parser.parse_args()
    if args.mode == "plot":
        plot_results(args.npz_file)
    elif args.mode == "gradient":
        run_gradient_viewer(args.npz_file)
    else:
        run_interactive_viewer(args.npz_file)


if __name__ == "__main__":
    main()
