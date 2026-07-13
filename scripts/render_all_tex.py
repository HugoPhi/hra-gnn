from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import fitz
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = ROOT / "reference_results"
BUILD_DIR = ROOT / "tmp" / "pdfs" / "all_latex_tables"
PDF_DIR = ROOT / "output" / "pdf" / "tables"
COMBINED_PDF = ROOT / "output" / "pdf" / "all_latex_tables.pdf"
IMAGE_DIR = ROOT / "doc" / "assets" / "tables"

# Every paper table fragment must be registered here. The coverage check below
# fails when a new .tex file is added without being included in the build.
TABLE_SOURCES = (
    "final_paper_auroc_ap_best",
    "paper_ablation",
    "main_table_first_all_models_best",
    "main_table_first_recent_best",
    "direct_baselines_diagnostic",
    "direct_baselines_two_datasets",
    "four_dataset_diagnostic",
    "model_complexity_theory",
    "model_parameters_tracelog",
)

ROTATED_TABLES = {
    "direct_baselines_diagnostic",
    "direct_baselines_two_datasets",
    "four_dataset_diagnostic",
}


def require_command(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"缺少命令 {name}，请先安装完整 LaTeX 环境。")


def validate_tex_coverage() -> None:
    artifact_dir = ROOT / "artifacts" / "results" / "tables"
    discovered = {path.stem for path in REFERENCE_DIR.glob("*.tex")}
    if artifact_dir.exists():
        discovered.update(path.stem for path in artifact_dir.glob("*.tex"))
    registered = set(TABLE_SOURCES)
    missing = discovered - registered
    nonexistent = registered - discovered
    if missing:
        raise RuntimeError(
            "发现尚未加入统一构建的 TeX 文件："
            + ", ".join(sorted(f"{name}.tex" for name in missing))
        )
    if nonexistent:
        raise RuntimeError(
            "统一构建清单引用了不存在的 TeX 文件："
            + ", ".join(sorted(f"{name}.tex" for name in nonexistent))
        )


def write_master_tex() -> Path:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    inputs = "\n\\clearpage\n".join(
        f"\\input{{reference_results/{name}.tex}}" for name in TABLE_SOURCES
    )
    source = (
        "\\documentclass[UTF8]{ctexart}\n"
        "\\usepackage[a4paper,margin=18mm]{geometry}\n"
        "\\usepackage{booktabs}\n"
        "\\usepackage{graphicx}\n"
        "\\usepackage{multirow}\n"
        "\\usepackage{rotating}\n"
        "\\usepackage{tabularx}\n"
        "\\pagestyle{empty}\n"
        "\\begin{document}\n"
        f"{inputs}\n"
        "\\end{document}\n"
    )
    master = BUILD_DIR / "all_latex_tables.tex"
    master.write_text(source, encoding="utf-8")
    return master


def compile_latex(master: Path) -> Path:
    require_command("latexmk")
    require_command("xelatex")
    command = [
        "latexmk",
        "-xelatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-outdir={BUILD_DIR}",
        str(master),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    pdf_path = BUILD_DIR / "all_latex_tables.pdf"
    if not pdf_path.exists():
        raise RuntimeError(f"LaTeX 编译结束，但没有生成 {pdf_path}")
    return pdf_path


def content_rect(page: fitz.Page, scale: float = 2.0) -> fitz.Rect:
    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    pixels = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, pixmap.n
    )
    ink = np.any(pixels[:, :, :3] < 245, axis=2)
    rows, columns = np.where(ink)
    if rows.size == 0:
        raise RuntimeError(f"第 {page.number + 1} 页为空，无法裁剪。")

    padding_pixels = 30
    x0 = max(int(columns.min()) - padding_pixels, 0) / scale
    y0 = max(int(rows.min()) - padding_pixels, 0) / scale
    x1 = min(int(columns.max()) + padding_pixels + 1, pixmap.width) / scale
    y1 = min(int(rows.max()) + padding_pixels + 1, pixmap.height) / scale
    return fitz.Rect(x0, y0, x1, y1) & page.rect


def cropped_page(source: fitz.Document, page_number: int, clip: fitz.Rect) -> fitz.Document:
    target = fitz.open()
    page = target.new_page(width=clip.width, height=clip.height)
    page.show_pdf_page(page.rect, source, page_number, clip=clip)
    return target


def write_outputs(source_pdf: Path) -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    COMBINED_PDF.parent.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    source = fitz.open(source_pdf)
    if len(source) != len(TABLE_SOURCES):
        raise RuntimeError(
            f"预期 {len(TABLE_SOURCES)} 页，实际生成 {len(source)} 页。"
        )

    combined = fitz.open()
    for page_number, name in enumerate(TABLE_SOURCES):
        clip = content_rect(source[page_number])
        cropped = cropped_page(source, page_number, clip)
        if name in ROTATED_TABLES:
            cropped[0].set_rotation(90)
        individual_pdf = PDF_DIR / f"{name}.pdf"
        individual_pdf.unlink(missing_ok=True)
        cropped.save(individual_pdf, garbage=4, deflate=True)
        combined.insert_pdf(cropped)

        page = cropped[0]
        svg_path = IMAGE_DIR / f"{name}.svg"
        svg_path.write_text(
            page.get_svg_image(text_as_path=True),
            encoding="utf-8",
        )

        png_path = IMAGE_DIR / f"{name}.png"
        pixmap = page.get_pixmap(matrix=fitz.Matrix(3.0, 3.0), alpha=False)
        pixmap.save(png_path)

        if svg_path.stat().st_size < 1_000 or png_path.stat().st_size < 1_000:
            raise RuntimeError(f"{name} 的渲染结果异常小，可能为空。")
        print(
            f"[完成] {name}: "
            f"{page.rect.width:.1f}x{page.rect.height:.1f} pt, "
            f"PNG {pixmap.width}x{pixmap.height}"
        )
        cropped.close()

    COMBINED_PDF.unlink(missing_ok=True)
    combined.save(COMBINED_PDF, garbage=4, deflate=True)
    combined.close()
    source.close()
    print(f"[完成] 合并 PDF: {COMBINED_PDF.relative_to(ROOT)}")


def main() -> None:
    validate_tex_coverage()
    master = write_master_tex()
    pdf_path = compile_latex(master)
    write_outputs(pdf_path)


if __name__ == "__main__":
    main()
