from pathlib import Path


def generate_latex_figures(
    fig_directory: Path,
    output_file: Path,
    figures_prefix="figures",
    pattern="*.png",
    default_caption=None,
    cosine_count: int | None = None,
    add_clearpage_period: int | None = None,
):
    files = sorted(fig_directory.rglob(pattern))
    if cosine_count is not None:
        files = [f for f in files if f.name.count("cosine") == cosine_count]

    with output_file.open("w", encoding="utf-8", newline="\n") as f:
        lines = []
        for i, file in enumerate(files):
            if default_caption is None:
                caption = file.stem
            else:
                caption = default_caption
            label = file.stem.replace(".", "_")  # Avoid issues with periods
            lines.append(r"\begin{figure}[htbp]")
            lines.append(r"    \centering")
            lines.append(rf"    \includegraphics[width=1.0\linewidth]{{{figures_prefix}/{file.name}}}")
            lines.append(r"    \caption*{" + caption + r"}")  # caption* means it won't show up in the list of figures
            lines.append(rf"    \label{{fig:{label}}}")
            lines.append(r"\end{figure}")  # Ensure double newline for spacing
            if add_clearpage_period is not None and i % add_clearpage_period == 0:
                lines.append(r"\clearpage")
        f.writelines("\n".join(lines) + "\n")


# Example usage


if __name__ == "__main__":
    # directory = Path(r"C:\Users\Colin\data\similar_but_not_the_same\embedding_analysis\score_analysis\plots")
    directory = Path.cwd() / "pose_evaluation/evaluation/plots"

    generate_latex_figures(
        directory,
        directory / "figures.tex",
        figures_prefix="figures/metrics_by_performance",
        default_caption=None,
        pattern="*.png",
    )
