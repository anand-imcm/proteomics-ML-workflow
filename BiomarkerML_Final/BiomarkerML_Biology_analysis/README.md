# PD–ALS downstream biological analysis

This standalone R project interprets **frozen BiomarkerML outputs** for a Nature Computational Science analysis. It does not modify BiomarkerML, retrain models, redefine feature selection, optimize biomarker thresholds, query STRING, or perform PPI analysis. The authoritative inclusion rule is always `Recommended == TRUE`; `RecommendedBiomarkerScore` is read directly from upstream output and a missing value remains `NA`.

The three analyses are fixed as PD versus HC, ALS versus HC, and `CrossDisease = PD versus ALS` (binary). A PD/ALS/HC multiclass analysis is not used to define cross-disease membership.

## Inputs and execution

Place immutable upstream files at:

```text
input/PD/recommended_biomarkers.csv
input/ALS/recommended_biomarkers.csv
input/CrossDisease/recommended_biomarkers.csv
input/expression_matrix.csv
```

Each biomarker CSV requires `Protein`, `Recommended`, and `RecommendedBiomarkerScore`. The expression file may be long (`SampleID`, `Protein`, `Abundance`, `Group`) or wide (`SampleID`, `Group`, then protein columns). Column names and the PD, ALS, and control labels are configured in `config/analysis_config.yaml`. Separate controls such as `HC_PD` and `HC_ALS` are supported and are never silently pooled. Missing configured groups cause a clear failure.

Run the production workflow from the project root:

```sh
Rscript scripts/06_generate_all.R
```

Run unit and isolated synthetic integration tests, then regenerate visual-QA figures:

```sh
Rscript tests/run_tests.R
Rscript tests/generate_test_figures.R
```

Synthetic data are confined to `tests/fixtures`, `tests/output`, and `figures/test`. The integration test copies the project to a temporary directory, enables an explicitly labelled deterministic fixture-only enrichment branch, and runs the same `Rscript scripts/06_generate_all.R` command. It never writes fixture results into production result paths.

## Protein universes and effect sizes

Two data-derived universes are retained:

- `MeasuredCommonUniverse`: every protein with at least one finite measurement in the harmonized analyzed expression data. Identifier-mapping failure does not remove a protein from this universe, effect-size calculation, or the X-C background.
- `EnrichmentUniverse`: the measured universe intersected with proteins successfully mapped to the enrichment identifier system. This measured-proteome universe—not the whole human genome—is used for every GO Biological Process, Reactome, and optional KEGG ORA.

The pipeline reports both sizes, mapped/unmapped counts, and protein-level mapping provenance.

For every measured protein, `R/effect_sizes.R` calculates PD-versus-control and ALS-versus-control Hedges' g. Positive values mean higher abundance in disease. For independent groups, with `df = n_disease + n_control - 2`, the implementation uses

```text
s_p^2       = ((n_disease - 1)s_disease^2 + (n_control - 1)s_control^2) / df
d           = (mean_disease - mean_control) / s_p
J           = 1 - 3/(4*df - 1)
g           = J*d
Var(g)      = J^2 * ((n_disease + n_control)/(n_disease*n_control) + d^2/(2*df))
SE(g)       = sqrt(Var(g))
CI(g)       = g ± z_(1-alpha/2) * SE(g)
```

The default CI is 95%. Insufficient sample size, zero/non-finite pooled SD, or another non-estimable case produces `NA` effect, variance, SE, and CI plus an explicit status; analytical effects are never `Inf` or `NaN`. Effect size characterizes frozen biomarkers and never changes membership.

## Cross-disease taxonomy

Predictive membership and measured abundance direction are separate variables.

`BiomarkerClass` is the authoritative, mutually exclusive predictive taxonomy: `Shared`, `Shared_differential`, `PD_selective`, `ALS_selective`, `PD_distinct`, `ALS_distinct`, or `Differential_only`. Raw `PD`, `ALS`, and `CrossDisease` booleans and `AnalysisSupport` are preserved. “Selective” describes model membership and is not a claim of disease-specific biology.

`DirectionClass` is independently derived from the signs of estimable PD and ALS Hedges' g values: `Concordant_up`, `Concordant_down`, either discordant orientation, or `Direction_uncertain` when an effect is missing or exactly zero. No magnitude, CI, or P-value threshold redefines `BiomarkerClass`.

`LegacyMembershipClass` is retained only as a deprecated compatibility/audit output. It is not propagated to current score matrices and is not used by enrichment, pathway convergence, figures, or any production scientific decision.

## Enrichment and figures

ORA uses identical EnrichmentUniverse, BH correction, database settings, and Jaccard-based redundancy reduction across comparisons. Classes below the configured mapped-protein minimum are reported as ineligible and return informative empty results without fabricated pathways.

Figure X asks how disease-level functional programs, biomarker-to-pathway composition, and measured molecular effects relate:

- X-A: comparative PD/ALS enrichment.
- X-B: disease-level recommended biomarker-to-pathway annotation membership.
- X-C: PD–ALS effect concordance for all measured proteins with estimable Hedges' g; frozen recommended biomarkers are overlaid. Spearman rho, two-sided P, and n are exported, but a proteome-wide P value is not interpreted as effect importance.

The former within-PD and within-ALS Spearman heatmaps remain analytical code and are exported as supplementary figures and source tables.

Figure Y asks how frozen predictive membership, ML evidence strength, revised functional architecture, and pathway convergence relate:

- Y-A: true PD/ALS/CrossDisease UpSet.
- Y-B: original upstream score heatmap with explicit `NA`, plus BiomarkerClass and DirectionClass annotations.
- Y-C: eligible BiomarkerClass ORA matrix.
- Y-D: deterministic protein-to-pathway annotation architecture for pathways supported on both disease axes.

For Y-D, the PD set contains all annotated recommended proteins with `PD == TRUE`, and the ALS set contains all with `ALS == TRUE`; shared proteins can occur on both sides. For each shared pathway, `ProteinJaccard = |P_PD ∩ P_ALS| / |P_PD ∪ P_ALS|` is exported as a continuous descriptive metric. Higher values indicate more overlap in supporting components and lower values indicate convergence through more disease-distinct components. It is not thresholded and does not imply causality. Edges represent annotation membership only.

## Outputs

Principal new analytical outputs are:

```text
results/measured_common_universe.csv
results/enrichment_universe.csv
results/protein_universe_summary.csv
results/effect_sizes/all_protein_effect_sizes.csv
results/effect_sizes/recommended_biomarker_effect_sizes.csv
results/biomarker_sets/revised_biomarker_taxonomy.csv
results/cross_disease/pathway_convergence_metrics.csv
results/pathway_mapping/YD_convergence_edges.csv
```

Complete and non-redundant enrichment tables, pathway links, display-selection audits, correlation tables, effect-concordance plot data/statistics, manifests, package versions, and QC reports are retained. Every primary panel and assembled figure is exported as 600-dpi PNG, LZW TIFF, and vector PDF at no more than 180 mm assembled width. Supplementary correlation figures are written under `figures/supplementary/` in the same formats.

Ranked enrichment, cell-type enrichment, genetics/GWAS, pQTL/colocalization, and external validation are out of scope. After a real-data run, inspect the assembled PNGs manually; automated QC cannot establish manuscript readability or absence of every collision.
