#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trace_module_guide.py
Generate self-documenting, schema-guided module reports.
- Stores module overviews + user-facing methods.
- Accepts discovery via regex OR a full grep command you provide.
- Extracts and organizes matches into our schema (constructor/inference/knobs/pre-post/writes).
- Emits clean Markdown you can cite in your paper.

Usage (examples at the bottom):
  python trace_module_guide.py --module "Signal Preprocessing" --discover-regex 'normalize|DAPI|area|size'
  python trace_module_guide.py --module "Clustering" --discover-regex 'scanpy|leiden|umap' --max-files 6
  python trace_module_guide.py --module "Spillover Compensation" --discover-cmd "grep -RIlE 'spillover|unmix|compensat' src/spacec"

Output:
  ./reports/SPACEc__<Module_Name>__guide_<timestamp>.md
"""

from __future__ import annotations
import argparse
import os
import re
import shlex
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# 1) Canonical module narratives (SPACEc)
#    Edit here to tweak wording or add more modules/tools later.
# ──────────────────────────────────────────────────────────────────────────────

MODULES: Dict[str, Dict] = {
    "Signal Preprocessing": {
        "overview": (
            "Signal preprocessing prepares single-cell intensities for analysis. "
            "SPACEc filters debris/artefacts by DAPI intensity and area, normalizes marker intensities to a comparable scale, "
            "removes noisy cells, and (optionally) visualizes spatial size distributions for QC."
        ),
        "methods": [
            ("Filter cells by DAPI intensity and area",
             "Apply a DAPI threshold to remove dim artefacts; filter by area range to drop debris and merged objects."),
            ("Normalize data",
             "Standardize intensities (e.g., z-score, arcsinh/log variants) so expression is comparable across channels."),
            ("Remove noisy cells",
             "Flag or remove outliers after normalization to stabilize clustering and annotation."),
            ("Show spatial distribution for size (optional)",
             "Plot tissue coordinates colored by size to spot regional artefacts prior to downstream steps.")
        ],
    },
    "Fluorescence Spillover Compensation": {
        "overview": (
            "Compensation reduces cross-channel bleed-through so each marker reflects its own signal. "
            "Load the per-cell matrix, estimate/apply a compensation model, visualize pre/post effects, and save the corrected data."
        ),
        "methods": [
            ("Load data", "Read segmented per-cell intensities (columns = markers) into memory."),
            ("Run compensation", "Estimate channel cross-talk (e.g., a spillover/mixing matrix) and subtract it."),
            ("Visualize compensation results", "Compare pre/post (heatmaps/scatter) to confirm bleed-through collapses."),
            ("Save data frames for further processing", "Write the corrected matrix for clustering/annotation.")
        ],
    },
    "Clustering": {
        "overview": (
            "Unsupervised clustering groups cells by expression patterns. "
            "Typical flow: build k-NN graph, embed (UMAP), assign Leiden/Louvain labels with QC between rounds."
        ),
        "methods": [
            ("Subclustering round 1", "Build neighbors graph on normalized features; run first Leiden/Louvain pass."),
            ("First QC", "Inspect UMAP and marker patterns; remove dubious cells/markers and tune parameters."),
            ("Subclustering round 2", "Refine broad clusters by re-clustering subsets or adjusting k/resolution."),
            ("Subclustering round 3", "Optionally split subtle subtypes if biologically warranted."),
            ("Final QC", "Stability checks on UMAP; lock labels for downstream use.")
        ],
    },
    "ML-enabled Cell Type Annotation (STELLAR)": {
        "overview": (
            "Supervised annotation assigns cell types from learned patterns (e.g., SVMs or STELLAR). "
            "Define features, train or load a model, score cells, inspect thresholds, and visualize predictions in tissue space."
        ),
        "methods": [
            ("Data explanation", "Specify features/markers and expected labels; align scales with training recipe."),
            ("Training", "Fit SVM or STELLAR (or load pretrained); validate to ensure generalization."),
            ("Inspect results", "Review score distributions and adjust thresholds/unknown handling."),
            ("Single-cell visualization", "Render predicted labels on tissue coordinates for plausibility.")
        ],
    },
    "Cellular Neighborhood Analysis": {
        "overview": (
            "Neighborhood analysis quantifies microenvironments. "
            "For each cell/window, count nearby cell types, cluster composition vectors into CNs, "
            "and derive spatial context maps or ternary/barycentric plots."
        ),
        "methods": [
            ("Cellular neighborhood analysis", "Pick k or a radius; count types in each window; cluster to define CNs."),
            ("Spatial context maps", "Find dominant CN combinations across space and organize contexts hierarchically."),
            ("Barycentric coordinates plot", "Use three CNs as triangle vertices; plot membership fractions to visualize gradients/interfaces.")
        ],
    },
    "TissUUmaps for Interactive Visualization": {
        "overview": (
            "TissUUmaps provides GPU-accelerated, zoomable inspection of cells in situ. "
            "Export coordinates + labels/layers and open them in the viewer for interactive exploration."
        ),
        "methods": [
            ("Instructions", "Prepare exports (coords, labels, cluster IDs, optional layers) in TissUUmaps-friendly format."),
            ("Integrated use", "Launch TissUUmaps on the processed outputs to explore tissue-wide maps."),
            ("Interactive cat plot via the TissUUmaps viewer", "Use viewer widgets (selection + composition plots) to compare regions.")
        ],
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# 2) Schema regexes (you can tweak/extend)
# ──────────────────────────────────────────────────────────────────────────────

SCHEMA_REGEX = {
    # What gets built before the main call
    "constructor": re.compile(
        r"(from_pretrained|fit\(|compile\(|initialize|Config|Classifier|Regressor|Model|"
        r"NearestNeighbors|kneighbors|radius_neighbors|KDTree|BallTree|"
        r"nnls\(|lstsq\(|pinv\(|spillover|mix(ing)?_?matrix|graph|matrix)",
        re.I
    ),
    # Where the work actually happens
    "inference": re.compile(
        r"(predict(_proba)?\(|transform\(|leiden\(|louvain\(|umap\(|cluster\(|"
        r"compensat|unmix|normalize\(|export|to_geojson|tmjson|write_(h5ad|zarr)|to_(csv|parquet|json))",
        re.I
    ),
    # Tunable parameters you can cite
    "knobs": re.compile(
        r"(n_neighbors|radius|metric|resolution|min_dist|cofactor|threshold|"
        r"alpha|lambda|min_size|max_size|n_pcs|n_components|\bkNN\b|\bk\b)",
        re.I
    ),
    # Anything right before/after that changes inputs/outputs
    "prepost": re.compile(
        r"(arcsinh|asinh|log1p|scale|z[-_]?score|winsor|clip|astype|rescale|filter|mask|"
        r"background|flatfield|gaussian|median|non[- _]?local[- _]?means|nlm)",
        re.I
    ),
    # What downstream consumers will read
    "writes": re.compile(
        r"(adata\.(obs|obsm|uns)\[|write_(h5ad|zarr)|to_(csv|parquet|json)|geojson|tmjson)",
        re.I
    ),
}

ENTRY_HINTS = ["run", "normalize", "compensate", "cluster", "infer_cell_types",
               "neighborhood", "export", "main"]


# ──────────────────────────────────────────────────────────────────────────────
# 3) Helpers
# ──────────────────────────────────────────────────────────────────────────────

def run_cmd(cmd: str) -> Tuple[int, str, str]:
    """Run a shell command; return (code, stdout, stderr)."""
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    return proc.returncode, out, err

def discover_files_via_cmd(discover_cmd: str, limit: int) -> List[Path]:
    """Accept a user-supplied grep command; parse file paths from its stdout."""
    code, out, err = run_cmd(discover_cmd)
    lines = out.strip().splitlines()
    files: List[Path] = []
    for line in lines:
        # Common grep formats: path, or path:line:col, or path:line
        p = Path(line.split(":")[0])
        if p.exists() and p.suffix in {".py", ".ipynb"}:
            files.append(p)
    # dedupe and cap
    uniq = []
    seen = set()
    for f in files:
        if f not in seen:
            uniq.append(f)
            seen.add(f)
    return uniq[:limit]

def discover_files_via_regex(root: Path, regex: str, limit: int) -> List[Path]:
    """Search for files containing regex (Python .py only for now)."""
    # Use grep if available for speed; else Python fallback.
    cmd = f"grep -RIlE --include='*.py' -i {shlex.quote(regex)} {shlex.quote(str(root))}"
    code, out, err = run_cmd(cmd)
    files = [Path(p) for p in out.strip().splitlines() if p]
    return files[:limit]

def read_file(path: Path) -> List[str]:
    try:
        return path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return []

def list_functions(src_lines: List[str]) -> List[Tuple[int, str]]:
    """Return [(lineno, func_name), ...] for defs."""
    out = []
    for i, line in enumerate(src_lines, start=1):
        m = re.match(r"^[ \t]*def[ \t]+([A-Za-z_][A-Za-z0-9_]*)\(", line)
        if m:
            out.append((i, m.group(1)))
    return out

def choose_entry_function(funcs: List[Tuple[int, str]], hint: Optional[str]) -> Optional[Tuple[int, str]]:
    # Priority: hint match, else first in ENTRY_HINTS, else first function
    names = [name for _, name in funcs]
    if hint and hint in names:
        return next((ln, nm) for ln, nm in funcs if nm == hint)
    for h in ENTRY_HINTS:
        if h in names:
            return next((ln, nm) for ln, nm in funcs if nm == h)
    return funcs[0] if funcs else None

def extract_function_body(lines: List[str], func_name: str) -> Tuple[int, List[str]]:
    """Return start line and body lines until next def or EOF."""
    start = None
    for i, line in enumerate(lines):
        if re.match(rf"^[ \t]*def[ \t]+{re.escape(func_name)}\(", line):
            start = i
            break
    if start is None:
        return (0, [])
    body = [lines[start]]
    for j in range(start + 1, len(lines)):
        if re.match(r"^[ \t]*def[ \t]+[A-Za-z_][A-Za-z0-9_]*\(", lines[j]):
            break
        body.append(lines[j])
    return (start + 1, body)  # 1-based for display

def grep_sections(lines: List[str]) -> Dict[str, List[Tuple[int, str]]]:
    """Return schema hits: dict cat -> [(lineno, line), ...]."""
    hits = {k: [] for k in SCHEMA_REGEX.keys()}
    for i, line in enumerate(lines, start=1):
        for cat, rx in SCHEMA_REGEX.items():
            if rx.search(line):
                hits[cat].append((i, line.rstrip()))
    return hits

def summarize_schema_as_paragraph(hits: Dict[str, List[Tuple[int, str]]]) -> str:
    """Convert schema hits into a brief chronological narrative."""
    parts = []
    def pick(cat, label):
        if hits.get(cat):
            lines = "; ".join([f"L{ln}" for ln, _ in hits[cat][:6]])
            parts.append(f"{label} (see {lines}).")
    pick("constructor", "Start by building the required object(s) or matrices")
    pick("inference",   "Next, apply the core operation to your data")
    pick("knobs",       "Tune key parameters to adjust behavior and resolution")
    pick("prepost",     "Use lightweight transforms before/after to stabilize inputs/outputs")
    pick("writes",      "Finally, persist outputs where downstream consumers will read them")
    return " ".join(parts) if parts else "No schema anchors detected in this file."

def fence(kind: str, text: str) -> str:
    return f"```{kind}\n{text.rstrip()}\n```\n" if text.strip() else ""

# ──────────────────────────────────────────────────────────────────────────────
# 4) Report writer
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Report:
    module: str
    overview: str
    methods: List[Tuple[str, str]]
    inputs_desc: str = ""
    outputs_desc: str = ""
    outdir: Path = Path("reports")
    fname_suffix: str = "guide"
    lines: List[str] = field(default_factory=list)

    def header(self):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.lines += [
            f"# SPACEc — {self.module}",
            f"_Report generated: {ts}_",
            "",
            "## What this module does (overview)",
            self.overview.strip(),
            "",
            "## What you can do (methods)",
        ]
        for name, desc in self.methods:
            self.lines.append(f"- **{name}.** {desc}")
        if self.inputs_desc or self.outputs_desc:
            self.lines += [
                "",
                "## Inputs & outputs at a glance",
                f"- **Inputs:** {self.inputs_desc or '—'}",
                f"- **Outputs:** {self.outputs_desc or '—'}",
            ]
        self.lines.append("\n---\n")

    def add_entry_file(self, idx: int, path: Path, entry_func: Optional[Tuple[int, str]],
                       body: List[str], hits: Dict[str, List[Tuple[int, str]]]):
        self.lines += [f"## Entry file [{idx}]: `{path}`", ""]
        if entry_func:
            ln, fn = entry_func
            self.lines += [f"**Chosen entry function:** `{fn}` (starts at L{ln}).", ""]
            snippet = "\n".join([f"{i:6d}  {line}" for i, line in enumerate(body, start=entry_func[0])])
            self.lines.append("**Function body**")
            self.lines.append(fence("text", snippet))
        else:
            self.lines.append("_No function definitions found; using file-level schema scan._\n")

        # Schema sections with a short narrative paragraph
        narrative = summarize_schema_as_paragraph(hits)
        self.lines += ["**Rebuild sequence (schema-guided):**", narrative, ""]
        for cat, title in [
            ("constructor", "CONSTRUCTOR / SETUP — builds objects/matrices/models/graphs"),
            ("inference",   "INFERENCE — apply the operation (segment/cluster/transform/predict/export)"),
            ("knobs",       "KNOBS — tunable parameters to cite"),
            ("prepost",     "PRE/POST — transforms right before/after the core call"),
            ("writes",      "WRITE POINTS — persisted outputs downstream will read"),
        ]:
            sect_hits = hits.get(cat) or []
            self.lines.append(f"### {title}")
            if not sect_hits:
                self.lines.append("_no matches_")
            else:
                blob = "\n".join([f"L{ln:>5}  {txt}" for ln, txt in sect_hits[:60]])
                self.lines.append(fence("text", blob))
            self.lines.append("")

        self.lines.append("\n---\n")

    def footer(self):
        self.lines += [
            "## Provenance & notes",
            ("Summaries reflect SPACEc’s published workflow (preprocessing/normalization, spillover compensation, "
             "clustering, ML annotation including STELLAR, cellular neighborhoods, and TissUUmaps integration). "
             "This guide is auto-assembled from concrete line-numbered matches to keep it faithful and reproducible."),
            ""
        ]

    def write(self) -> Path:
        self.outdir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        name = f"SPACEc__{self.module.replace(' ', '_')}__{self.fname_suffix}_{ts}.md"
        path = self.outdir / name
        path.write_text("\n".join(self.lines), encoding="utf-8")
        return path


# ──────────────────────────────────────────────────────────────────────────────
# 5) Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Schema-guided module tracer with narrative output.")
    ap.add_argument("--module", required=True, help="Module name (must exist in MODULES dict).")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--discover-regex", help="Regex to find candidate files (searched recursively).")
    grp.add_argument("--discover-cmd", help="Full shell grep command returning matching files (paths may be colon-suffixed).")
    ap.add_argument("--src", default=None, help="Source dir (default: src/spacec if exists, else repo root).")
    ap.add_argument("--max-files", type=int, default=5, help="Cap on entry files to parse.")
    ap.add_argument("--entry-hint", default=None, help="Prefer this function as entry if present (e.g., run, cluster, normalize).")
    ap.add_argument("--inputs", default="", help="Optional: free text for inputs line.")
    ap.add_argument("--outputs", default="", help="Optional: free text for outputs line.")
    args = ap.parse_args()

    # Resolve module narrative
    if args.module not in MODULES:
        print(f"[error] Unknown module: {args.module}. Known: {', '.join(MODULES.keys())}", file=sys.stderr)
        sys.exit(1)
    meta = MODULES[args.module]

    # Resolve root + src
    try:
        root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    except Exception:
        root = Path.cwd()
    src = Path(args.src) if args.src else (root / "src" / "spacec" if (root / "src" / "spacec").exists() else root)

    # Discover candidate files
    if args.discover_cmd:
        files = discover_files_via_cmd(args.discover_cmd, args.max_files)
    else:
        files = discover_files_via_regex(src, args.discover_regex, args.max_files)

    if not files:
        print("[warn] No candidate files found. Check your regex/command or adjust --src.", file=sys.stderr)

    # Build report
    rpt = Report(
        module=args.module,
        overview=meta["overview"],
        methods=meta["methods"],
        inputs_desc=args.inputs,
        outputs_desc=args.outputs,
    )
    rpt.header()

    for i, fpath in enumerate(files, start=1):
        lines = read_file(fpath)
        funcs = list_functions(lines)
        entry = choose_entry_function(funcs, args.entry_hint)
        if entry:
            _, fn = entry
            start_ln, body = extract_function_body(lines, fn)
            # Use full-file schema matches; keeps it simple (entry + helpers often share file)
            hits = grep_sections(lines)
            rpt.add_entry_file(i, fpath, (start_ln, fn), body, hits)
        else:
            hits = grep_sections(lines)
            rpt.add_entry_file(i, fpath, None, [], hits)

    rpt.footer()
    path = rpt.write()
    print(f"→ guide written: {path}")

if __name__ == "__main__":
    main()
