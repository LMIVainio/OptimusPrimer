# OptimusPrimer
Primer design tool
This is OptimusPrimer, a comprehensive Python-based tool for designing and optimizing PCR primers for DNA amplification and cloning. It focuses on finding optimal forward and reverse primer pairs that minimize secondary structures, dimers, and other thermodynamic issues while meeting user-defined constraints.

## Key Features
- Primer Design: Automatically searches for primers in specified regions of a DNA sequence, expanding window sizes to meet melting temperature thresholds.
-Scoring System: Evaluates primers based on GC content, GC clamps, melting temperature deviations, secondary structure stability (MFE), and product size.
-Secondary Structure Analysis: Uses ViennaRNA to calculate minimum free energies (MFE) for primer hairpins, homodimers, and heterodimers, including adapter effects.
- Visualization: Plots primer dimers and linear sequences with color-coded nucleotides and adapters. Visualizes sequence annotations with optional score overlays.
- Adapter Support: Integrates cloning adapters (e.g., BsaI, Gateway) and accounts for their impact on primer thermodynamics.
- Constraint-Based Filtering: Configurable thresholds for length, Tm, MFE, product size, etc., with presets for different adapter sets.
- Sequence Annotation: Highlights promoters, CDS, search regions, and restriction sites (e.g., BsaI) in HTML visualizations.
- Primer3 Compatibility: Outputs sequences in Primer3 annotation format for benchmarking
- Batch Analysis: Processes primers from Excel files, analyzing sequences with adapters.
## Requirements
- Python Libraries:
  - viennarna (via conda: `conda install bioconda::viennarna` or pip: `pip install viennarna`)
  - `biopython` (pip: `pip install biopython`)
  - `matplotlib`, `numpy`, `pandas` (standard Anaconda/Miniconda installs)
- Jupyter Notebook for interactive execution.
- Optional: Excel files for batch primer analysis.

## Usage Overview
1. Setup: Define your DNA sequence (e.g., promoter + CDS), search regions for forward/reverse primers, and adapter set (e.g., 'BsaI_1', 'GW_box1', 'None' or your custom adapters).
![alt text](image/Picture1.png)
 
Promoter: annotated promoter region, F search region: specified search area for forward primers, BsaI cut sites visualized, R search region: specified search are for reverse primers, CDS: annotated coding sequence. HTML based annotations.
2. Find Primers:
   - Use `find_primers()` to search for candidates in forward/reverse orientations.
   - Adjust constraints (e.g., Tm threshold, product size) via the `CONSTRAINTS` dict.
3. Pair and Score:
   - `find_primer_pairs()` combines forward and reverse candidates, filtering by Tm match, product size, and MFE.
   - `score_primers()` ranks pairs using weighted criteria (GC clamp, Tm deviation, MFE, etc.).
 
 
Table of primers with Forward/Reverse pairs with scoring
4. Visualize and Validate:
   - `plot_dimer()` and `plot_linear()` for primer structure visualization.
   - `visualize_sequence_html()` for annotated sequence display with score heatmaps.
   - Check for overlaps or issues with `remove_overlapping_primers()`.
 
 
linear primers with adapters
 
Primer secondary structures (this is heterodimer example)
5. Batch Processing:
   - `analyze_primers_from_xlsx()` reads primers from Excel, analyzes them, and exports results.

## Example Workflow

- Load a sequence (promoter + coding region).
- Define search regions (e.g., forward in promoter, reverse near CDS end).
- Select adapters and constraints.
- Run primer search and pairing.
- Visualize top pairs and their dimers.
- Export or refine based on scores.

## Key Functions

- `find_primers()`: Sliding-window search for primers meeting Tm and MFE thresholds.
- `find_primer_pairs()`: Pairs forward/reverse primers, computes heterodimers, and filters.
- `score_primers()`: Scores pairs with penalties for deviations from optima.
- `plot_dimer()`: Plots RNA secondary structures for dimers.
- `visualize_sequence_html()`: HTML visualization of sequences with annotations and scores.
- `analyze_primer()`: Analyzes a single primer for Tm, GC, MFE, etc.

## Notes
- Designed for cloning workflows (e.g., Golden Gate with BsaI sites).
- Handles reverse complement for reverse primers.
- Future enhancements: BLAST integration, Tm with mismatches, adapter optimization.
- Run cells sequentially in Jupyter for interactive design.

For detailed code comments and parameters, refer to the notebook cells. This tool streamlines primer design for molecular biology applications.
