# IndependentStudy
This repository accompanies my report, “Independent Study – Spatial Proteomics Pipelines and Workflows.” The paper reviews four toolchains—SPEX, SPACEc, ScProAtlas, and SpatialProteomics—and documents their core modules, algorithms, and expected inputs/outputs. The repo hosts minimal, tool-scoped source snapshots that the paper cites, so readers can jump from a method described in the manuscript to the exact code path here.

### What's here (repo layout → report sections)

```
├─ SPEX/ # Source snapshot for SPEX
│    ├─ spex_segmentation/ # Segmentation and post-processing modules
│    └─ spex_spatial_transcriptomics/ # Spatially informed transcriptomics modules
├─ SPACEc/ # SPACEc analysis notebooks/scripts (image→features→CN→proximity)
├─ spatialproteomics/ # Python package used in pipelines (plots, stats, CN tools, etc.)
└─ external/
     └─ scProAtlas_analysis/ # Selected ScProAtlas analysis scripts (see note below)
```

   
**How the repo maps to the paper**

- **SPEX — see “SPEX Workflow Analysis” (p. 8).**<br>
The code here backs image pre-processing, segmentation, niche/neighborhood analysis,<br>
and expression/pathway modules referenced in the IMC tonsil study example.

- **SPACEc — supports “SPACEc Modules of Interest” (p. 15).**<br>
Line Covers tissue detection, segmentation (Cellpose/DeepCell), per-cell table assembly & normalization, clustering/label transfer, and the platform’s patch-proximity readouts.

- **ScProAtlas (external) — supports the “ScProAtlas” section (p. 22).**<br>
As explained in the paper, ScProAtlas is primarily a website + curated database with results hosted externally; its public codebase is not a clean, importable Python package. I’ve included representative analysis scripts here for traceability (e.g., CellPhoneDB interaction runs, Moran’s-I spatially variable features, and general utilities) even though most logic lives in web services and precomputed tables.

- **SpatialProteomics (library) — supports the “SpatialProteomics” section (p. 24).**<br>
Used throughout for plotting, neighborhood/graph computations, and diversity/statistical<br>
summaries as described in “SpatialProteomics.”

❗️IMPORTANT: Each section in the paper that describe a module and its methods/functions point to folder/file paths exactly as they appear here, so you can cross-check a described method with its implementation.

**How to use this repository while reading the paper**
1. Start with the paper (root of repo) to understand each tool’s pipeline stages and outputs. When a section references a file path, open the matching path in this repo to see the code in context. 
2. Browse by tool:
   - SPEX → SPEX/…<br>
   - SPACEc → SPACEc/…<br>
   - SpatialProteomics → spatialproteomics/…<br>
   - ScProAtlas → external/scProAtlas_analysis/… (see limitations below)<br>
3. Focus on interfaces: In some cases, the paper emphasizes function signatures (“inputs / knobs / outputs”) and where artifacts are written (tables, AnnData fields, obsm/obsp keys). Use those cues to locate the exact call sites in each module.

**Reproducibility notes**
**Environment:** All exploration was performed in GitHub Codespaces to avoid local dependency drift and to keep paths stable with the manuscript’s citations. You can click the green Code ▾ → Create codespace to reproduce the same layout. 
**Data:** Example datasets used in vendor/validation demos (e.g., IMC ROIs) are not redistributed here. The code is only sufficient to understand algorithms and expected inputs/outputs.

**About ScProAtlas code isolation**
As the paper explains, ScProAtlas is delivered as a web application + curated database with precomputed results; much of the logic is embedded in the web stack or external notebooks rather than in a stable, importable Python library. That architecture limits clean module isolation compared to SPEX/SPACEc/SpatialProteomics.

Even so, I’ve committed relevant snapshot scripts under external/scProAtlas_analysis/ (e.g., CellPhoneDB batch runs; Moran’s-I spatially variable gene/protein calls; small utilities) so readers can see exactly what was available and how it was invoked. The primary analyses in ScProAtlas are best understood through the paper’s descriptions and the linked code paths here rather than through a uniform Python API.

**How to run code (optional)**
  This repo is documentation-first. If you want to run parts of it:
  1. Use a fresh Codespace obtain directly from the original repo (this repo serves as isolated snapshots) 
  2. Install per-tool dependencies as needed (e.g., scanpy, anndata, squidpy, cellpose, pandas, numpy, matplotlib,  seaborn). Some vendor pipelines assume GPU or large-RAM environments.
  3. Follow instructions as provided by the original manual/codesource documentation 

**Attribution**<br>
This repository analyzes parts of the following tools. Please credit the original authors when using methods derived from their work:<br>
SPEX (Spatial Expression Explorer) — Xiao Li, Ximo Pechuan-Jorge, Tyler Risom, Conrad Foo, Alexander Prilipko, Artem Zubkov, Caleb Chan, Patrick Chang, Frank Peale, James Ziai, Sandra Rost, Derrek Hibar, Lisa McGinnis, Evgeniy Tabatsky, Xin Ye, Hector Corrada Bravo, Zhen Shi, Malgorzata Nowicka, Jon Scherdin, James Cowan, Jennifer Giltnane, Darya Orlova, Rajiv Jesudason. doi: <br>
https://doi.org/10.1101/2022.08.22.504841<br>

SPACEc (Spatial Analysis for Codex Exploration) — Yuqi Tan, Tim N. Kempchen, Martin Becker, Maximilian Haist, Dorien Feyaerts, Yang Xiao, Graham Su, Andrew J. Rech, Rong Fan, John W. Hickey, Garry P. Nolan. doi: https://doi.org/10.1101/2024.06.29.601349<br>

ScProAtlas — Tiangang Wang, Xuanmin Chen, Yujuan Han, Jiahao Yi, Xi Liu, Pora Kim, Liyu Huang, Kexin Huang, Xiaobo Zhou. Nucleic Acids Research. DOI: 10.1093/nar/gkae990 (PMID: 39526396, PMCID: PMC11701751).<br>

SpatialProteomics (Python library) — Matthias Fabian Meyer-Bender, Harald Sager Voehringer, Christina Schniederjohann, Sarah Patricia Koziel, Erin Kim Chung, Ekaterina Popova, Alexander Brobeil, Lisa-Maria Held, Aamir Munir, scverse Community, Sascha Dietrich, Peter-Martin Bruch, Wolfgang Huber. bioRxiv 2025.04.29.651202; doi: https://doi.org/10.1101/2025.04.29.651202<br>

**Note.** Any upstream files included here as snapshots (e.g., under external/…) retain their original licenses and copyright.


**Questions or issues?** Open a GitHub issue in this repo with the folder/file path you’re looking at and the section of the paper you’re mapping from.

