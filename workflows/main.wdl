version 1.0

import "./tasks/run_plan.wdl" as RP
import "./tasks/dim_reduction.wdl" as DR
import "./tasks/ml_general.wdl" as MLGEN
import "./tasks/ml_vae.wdl" as MLVAE
import "./tasks/summary.wdl" as SUMM
import "./tasks/pdf_report.wdl" as REP
import "./tasks/ml_reg.wdl" as MLREG
import "./tasks/protein_network.wdl" as PPI

workflow main {
    input {
        File input_csv
        String output_prefix
        String mode = "Summary" # choices: Classification, Regression, Summary
        String? dimensionality_reduction_choices # PCA ELASTICNET KPCA UMAP TSNE PLS
        Int number_of_dimensions = 3
        String classification_model_choices = "RF" # KNN NN SVM XGB PLSDA VAE LR GNB LGBM MLPVAE
        String regression_model_choices = "RF_reg" # NN_reg SVM_reg XGB_reg PLS_reg KNN_reg LGBM_reg VAE_reg MLPVAE_reg
        Boolean calculate_shap = false
        Boolean run_ppi = false
        Int shap_features = 10
    }
    String pipeline_version = "1.0.2"
    String container_gen = "ghcr.io/anand-imcm/biomarkerml-gen:~{pipeline_version}"
    String container_ppi = "ghcr.io/anand-imcm/biomarkerml-net:~{pipeline_version}"
    Array[File] default_arr = []
    call RP.run_plan {
        input: model_choices = classification_model_choices,
            dimensionality_reduction_choices = dimensionality_reduction_choices,
            regression_choices = regression_model_choices,
            mode = mode,
            shap = calculate_shap,
            ppi = run_ppi,
    }
    if (run_plan.use_dim){
        scatter (dim_method in run_plan.dim_opt) {
            call DR.dim_reduction {
                input:
                    input_csv = input_csv,
                    output_prefix = output_prefix,
                    dim_method = dim_method,
                    num_dimensions = number_of_dimensions,
                    docker = container_gen
            }
        }
    }
    Array[File] dim_out = if (run_plan.use_dim) then flatten(select_all([dim_reduction.out])) else default_arr
    Array[File] dim_png = if (run_plan.use_dim) then flatten(select_all([dim_reduction.png])) else default_arr
    if (run_plan.use_gen) {
        scatter (gen_method in run_plan.gen_opt) {
            call MLGEN.ml_gen {
                input:
                    model = gen_method,
                    input_csv = input_csv,
                    output_prefix = output_prefix,
                    dim_opt = run_plan.dim_opt[0],
                    docker = container_gen
            }
        }
    }
    if (run_plan.use_vae) {
        scatter (vae_method in run_plan.vae_opt) {
            call MLVAE.ml_vae {
                input:
                    model = vae_method,
                    input_csv = input_csv,
                    output_prefix = output_prefix,
                    dim_opt = run_plan.dim_opt[0],
                    docker = container_gen
            }
        }
    }
    Array[File] gen_ml_out = if (run_plan.use_gen) then flatten(select_all([ml_gen.data])) else default_arr
    Array[File] vae_ml_out = if (run_plan.use_vae) then flatten(select_all([ml_vae.data])) else default_arr
    Array[File] classification_out = flatten([gen_ml_out, vae_ml_out])
    if (run_plan.use_reg) {
        call MLREG.ml_reg {
            input:
                model = run_plan.reg_opt,
                input_csv = input_csv,
                output_prefix = output_prefix,
                dim_opt = run_plan.dim_opt[0],
                docker = container_gen
        }
    }
    Array[File] reg_out = if (run_plan.use_reg) then select_all([ml_reg.results]) else default_arr
    Array[String] model_opts = flatten([run_plan.gen_opt, run_plan.vae_opt, run_plan.reg_opt])
    Array[File] model_data = if (!run_plan.use_dim) then flatten([classification_out, reg_out]) else default_arr
    if (!run_plan.use_dim){
        call SUMM.summary {
            input:
                dataset = model_data,
                model = model_opts,
                output_prefix = output_prefix,
                use_shap = run_plan.use_shap,
                use_reg = run_plan.use_reg,
                shap_num_feat = shap_features,
                docker = container_gen
        }
    }
    if(run_plan.use_ppi){
        call PPI.ppi_analysis{
            input:
                summary_set = summary.results,
                proteinExpFile = input_csv,
                output_prefix = output_prefix,
                docker = container_ppi
        }
    }
    Array[File] summary_files = if (!run_plan.use_dim) then select_all([summary.results]) else default_arr
    Array[File] ppi_files = if (run_plan.use_ppi) then select_all([ppi_analysis.results]) else default_arr
    Array[File] all_results = flatten([dim_out, dim_png, summary_files, ppi_files])
    call REP.pdf_report {
        input:
            summary_set = all_results,
            output_prefix = output_prefix,
            docker = container_gen
    }
    
    output {
        File report = pdf_report.out
        File results = pdf_report.results
    }
}