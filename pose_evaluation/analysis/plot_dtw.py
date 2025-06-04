from itertools import product
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from fastdtw import fastdtw  # type: ignore (pylint can't find it but it's there)


def pad_seq(seq, val, count):
    seq_pad = [val for _ in range(count)]
    seq_pad.extend(seq)
    return seq_pad


def apply_minimal_layout(fig: go.Figure, is_3d: bool = True, size: int = 600) -> None:
    if is_3d:
        fig.update_layout(
            title=None,
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                bgcolor="white",
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
            paper_bgcolor="white",
            width=size,
            height=size,
        )
    else:
        fig.update_layout(
            title=None,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
            paper_bgcolor="white",
            plot_bgcolor="white",
            width=size,
            height=size,
        )


def export_minimal_pdf(fig: go.Figure, output_path: Path, is_3d: bool = True, size: int = 600) -> None:
    apply_minimal_layout(fig, is_3d=is_3d, size=size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pio.write_image(fig, str(output_path.with_suffix(".pdf")), format="pdf")
    print(f"Exported: {output_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    out_path = Path("dtw_plots")
    out_path.mkdir(exist_ok=True)

    # Generate angles from 0 to 2π
    point_count_1_list = [12, 6]
    paddings = ["padwithfirstframe", "zeropad", None]
    add_mappings = [
        True,
        # False
    ]
    add_z_values = [True, False]
    # add_z_values = [True]
    # add_z = True

    camxyz = (-2.0, -2.0, 1.0)
    POINT_COUNT_2 = 6

    update_x_vals = [True, False]

    for point_count_1, padding, add_mappings, add_z, update_x in product(
        point_count_1_list, paddings, add_mappings, add_z_values, update_x_vals
    ):

        print(
            f"Point Counts: {point_count_1},{POINT_COUNT_2}.",
            " Padding:{padding}, add_mappings: {add_mappings}, add_z: {add_z}, update_x: {update_x}",
        )

        TITLE = ""

        if point_count_1 == POINT_COUNT_2:
            TITLE = TITLE + f"Equal-Length Sequences ({point_count_1},{POINT_COUNT_2})"
            if padding is not None:
                print(f"SKIPPING! No point adding {padding} if they're the same length")
                continue

        else:
            TITLE = TITLE + f"Unequal-Length Sequences ({point_count_1},{POINT_COUNT_2})"
        MODE = "lines+markers"

        angles1 = np.linspace(0, 2 * np.pi, point_count_1)
        angles2 = np.linspace(0, 2 * np.pi, POINT_COUNT_2)

        # Compute sine and cosine values
        trace1_y = np.cos(angles1)
        # trace2_values = np.cos(angles2)
        # trace2_values = np.sin(angles2)
        trace2_y = np.cos(angles2 - np.pi / 8)  # Shift right by π/8

        if add_z:
            trace2_y = trace2_y + 5
        else:
            trace2_y = trace2_y + 1
        trace1_z = [i * 0.25 for i in range(len(trace1_y))]
        trace2_z = [i * 0.25 for i in range(len(trace2_y))]

        if padding == "zeropad":
            TITLE = f"{TITLE} (Zeropad)"
            pad_count = point_count_1 - POINT_COUNT_2
            PAD_VALUE = 0
            # trace2_y_pad = [0 for i in range(point_count_1 - point_count_2)]
            # trace2_y_pad.extend(trace2_y)
            trace2_y = pad_seq(trace2_y, val=PAD_VALUE, count=pad_count)

            # trace2_z_pad = [0 for i in range(point_count_1 - point_count_2)]
            # trace2_z_pad.extend(trace2_z)
            # trace2_z = trace2_z_pad
            trace2_z = pad_seq(trace2_z, val=PAD_VALUE, count=pad_count)

            if update_x:
                # s2_pad = [0 for i in range(point_count_1 - point_count_2)]
                # angles2_pad.extend(angles2)
                # angles2 = angles2_pad
                angles2 = pad_seq(angles2, val=PAD_VALUE, count=pad_count)
                TITLE += "(X also padded)"
            else:
                angles2 = angles1

        elif padding == "padwithfirstframe":
            TITLE = f"{TITLE} (Pad with First Frame)"
            pad_count = point_count_1 - POINT_COUNT_2
            # trace2_y_pad = [trace2_y[0] for i in range(point_count_1 - point_count_2)]
            # trace2_y_pad.extend(trace2_y)
            # trace2_y = trace2_y_pad
            trace2_y = pad_seq(trace2_y, val=trace2_y[0], count=pad_count)

            # trace2_z_pad = [trace2_z[0] for i in range(point_count_1 - point_count_2)]
            # trace2_z_pad.extend(trace2_z)
            # trace2_z = trace2_z_pad
            trace2_z = pad_seq(trace2_z, val=trace2_z[0], count=pad_count)

            # angles2_pad = [angles2[0] for i in range(point_count_1 - point_count_2)]
            # angles2_pad.extend(angles2)
            # angles2 = angles2_pad
            if update_x:
                # s2_pad = [0 for i in range(point_count_1 - point_count_2)]
                # angles2_pad.extend(angles2)
                # angles2 = angles2_pad
                angles2 = pad_seq(angles2, val=angles2[0], count=pad_count)
                TITLE += " (X also padded)"
            else:
                angles2 = angles1

        if len(trace1_y) == len(trace2_y):
            mappings = [(i, i) for i in range(point_count_1)]
        else:
            points1 = [xyz for xyz in zip(angles1, trace1_y, trace1_z)]
            points2 = [xyz for xyz in zip(angles2, trace2_y, trace2_z)]
            # dist, mappings = fastdtw(trace1_y, trace2_y, 1)
            # print(f"Points 1: {points1}")
            # print(f"Points 2: {points2}")
            dist, mappings = fastdtw(points1, points2, 1)
            TITLE = f"{TITLE} (DTW)"
        # print(mappings)
        # mode = "markers"

        # Create the plot
        latex_fig = go.Figure()

        if add_z:
            print("Adding the THIRD DIMENSION")
            # Add trace
            latex_fig.add_trace(
                go.Scatter3d(
                    x=angles1,
                    y=trace1_y,
                    z=trace1_z,
                    # z=[i for i in range(len(trace1_y))],
                    #  mode="lines",
                    mode=MODE,
                    marker=dict(size=2),
                    name=f"Sequence 1: {point_count_1} points",
                )
            )

            # Add other trace
            latex_fig.add_trace(
                go.Scatter3d(
                    x=angles2,
                    y=trace2_y,
                    z=trace2_z,
                    # z=[i for i in range(len(trace2_y))],
                    #  mode="lines",
                    mode=MODE,
                    marker=dict(size=2),
                    name=f"Sequence 2 {POINT_COUNT_2} points",
                )
            )
        else:
            # Add trace
            latex_fig.add_trace(
                go.Scatter(
                    x=angles1,
                    y=trace1_y,
                    mode=MODE,
                    marker=dict(size=2),
                    name=f"Sequence 1: {point_count_1} points",
                )
            )

            # Add other trace
            latex_fig.add_trace(
                go.Scatter(
                    x=angles2,
                    y=trace2_y,
                    mode=MODE,
                    marker=dict(size=4),
                    name=f"Sequence 2 {POINT_COUNT_2} points",
                )
            )

        # Add green lines for mappings
        if add_mappings:

            for i, j in mappings:
                if add_z:
                    latex_fig.add_trace(
                        go.Scatter3d(
                            x=[angles1[i], angles2[j]],
                            y=[trace1_y[i], trace2_y[j]],
                            z=[trace1_z[i], trace2_z[j]],
                            mode="lines",
                            line=dict(color="green", width=2),
                            showlegend=False,  # Hide individual mappings from legend
                        )
                    )
                else:
                    latex_fig.add_trace(
                        go.Scatter(
                            x=[angles1[i], angles2[j]],
                            y=[trace1_y[i], trace2_y[j]],
                            mode="lines",
                            line=dict(color="green", width=2),
                            showlegend=False,  # Hide individual mappings from legend
                        )
                    )

        # Customize layout

        if add_z:
            latex_fig.update_layout(scene=dict(camera=dict(eye=dict(x=camxyz[0], y=camxyz[1], z=camxyz[2]))))
            TITLE = f"{TITLE} (3D)"
            # title = f"{title} (3D, cam={camxyz})"

        if add_mappings:
            TITLE = f"{TITLE} (Mappings)"

        latex_fig.update_layout(
            title=TITLE,
            xaxis_title="Angle (radians)",
            yaxis_title="Value",
            legend_title="Functions",
            showlegend=True,
        )
        print(TITLE)

        # Show the plot
        # fig.show()
        fig_out = out_path / f"{TITLE}".replace(" ", "")
        latex_fig.write_html(f"{fig_out}.html")

        print(fig_out)
        # pio.write_image(fig, f"{fig_out}.pdf", format="pdf")
        export_minimal_pdf(latex_fig, fig_out, is_3d=add_z)
        print("*" * 60)
