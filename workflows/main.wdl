version 1.0

import "./tasks/preprocessing.wdl" as pre
import "./tasks/summary.wdl" as report
import "./dim_reduction.wdl" as drwf
import "./standard_ml.wdl" as mlwf

workflow main {
    input {
        File input_csv
        String output_prefix
        String model_choices = "RF"
        String dimensionality_reduction_choices = "PCA"
        Boolean use_dimensionality_reduction = false
        Boolean skip_ML_models = false
        Int shap_radar_num_features = 10
        Int num_of_dimensions = 3
    }
    String pipeline_version = "1.0.0"
    String container_gen = "docker.io/library/proteomics:~{pipeline_version}"
    String container_vae = "docker.io/library/vae:~{pipeline_version}"
    Array[File] default_arr = []
    if (use_dimensionality_reduction && skip_ML_models)  {
        call drwf.dim_reduction_wf as dim_reduction {
            input:
                input_csv = input_csv,
                output_prefix = output_prefix,
                docker = container_gen,
                method_name = dimensionality_reduction_choices,
                num_of_dimensions = num_of_dimensions
        } 
    }
    if (use_dimensionality_reduction && !skip_ML_models) {
        call drwf.dim_reduction_wf as dim_reduction_ml {
            input:
                input_csv = input_csv,
                output_prefix = output_prefix,
                docker = container_gen,
                method_name = dimensionality_reduction_choices,
                num_of_dimensions = num_of_dimensions
        } 
        call mlwf.standard_ml_wf as ml_dim {
            input:
                input_csv = dim_reduction_ml.csv,
                output_prefix = output_prefix,
                container_gen = container_gen,
                container_vae = container_vae,
                model_choices = model_choices,
                shap_num_features = shap_radar_num_features
                
        }
    }
    if (!use_dimensionality_reduction && skip_ML_models) {
        call pre.preprocessing_std as std_csv_def {
            input: 
                input_csv = input_csv,
                output_prefix = output_prefix,
                docker = container_gen
        }
    }
    if (!use_dimensionality_reduction && !skip_ML_models) {
        call pre.preprocessing_std as std_csv {
            input: 
                input_csv = input_csv,
                output_prefix = output_prefix,
                docker = container_gen
        }
        call mlwf.standard_ml_wf as ml_std {
            input:
                input_csv = std_csv.csv,
                output_prefix = output_prefix,
                container_gen = container_gen,
                container_vae = container_vae,
                model_choices = model_choices,
                shap_num_features = shap_radar_num_features
        }
    }
    File overall_roc_plots = if (!skip_ML_models) then select_first([
        ml_std.out_all_roc_curves,
        ml_dim.out_all_roc_curves,
    ]) else input_csv
    Array[File] dim_reduct_plots = if (use_dimensionality_reduction) then flatten(select_all([
        dim_reduction.png_list,
        dim_reduction_ml.png_list
    ])) else default_arr
    Array[File] confusion_matrix_plots = if (!skip_ML_models) then flatten(select_all([
        ml_std.out_cls_confusion_matrix_plot,
        ml_dim.out_cls_confusion_matrix_plot,
    ])) else default_arr
    Array[File] eval_matrix_plots = if (!skip_ML_models) then flatten(select_all([
        ml_std.out_cls_metrics_plot,
        ml_dim.out_cls_metrics_plot,
    ])) else default_arr
    Array[File] roc_curve_plots = if (!skip_ML_models) then flatten(select_all([
        ml_std.out_cls_roc_curve_plot,
        ml_dim.out_cls_roc_curve_plot,
    ])) else default_arr
    Array[File] shap_radar_plots = if (!skip_ML_models) then flatten(select_all([
        ml_std.out_radar_plot,
        ml_dim.out_radar_plot,
    ])) else default_arr
    Array[File] shap_csv_out = if (!skip_ML_models) then flatten(select_all([
        ml_std.out_shap_values,
        ml_dim.out_shap_values
    ])) else default_arr
    Array[File] all_valid_files = flatten([
        [overall_roc_plots],
        dim_reduct_plots,
        confusion_matrix_plots,
        eval_matrix_plots,
        roc_curve_plots,
        shap_radar_plots
    ])
    Array[File] dim_csv_output = if (use_dimensionality_reduction) then flatten(select_all([
        dim_reduction.csv_list,
        dim_reduction_ml.csv_list
        ])) else default_arr
    File std_csv_output =  if (!use_dimensionality_reduction) then select_first([
        std_csv_def.csv,
        std_csv.csv
        ]) else input_csv
    call report.summary as analysis_report {
        input:
            summary_data = all_valid_files,
            output_prefix = output_prefix,
            docker = container_gen
    }
    output {
        Array[File] dimensionality_reduction_csv = dim_csv_output
        Array[File] dimensionality_reduction_plots = dim_reduct_plots
        File std_preprocessing_csv = std_csv_output
        Array[File] shap_csv = shap_csv_out
        File report = analysis_report.report
        File plots = analysis_report.plots
    }
}