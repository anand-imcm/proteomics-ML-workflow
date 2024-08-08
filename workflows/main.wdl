version 1.0

import "./tasks/preprocessing.wdl" as pre
import "./tasks/classification.wdl" as cls
import "./tasks/summary.wdl" as report

workflow main {
    input {
        File input_csv
        String output_prefix
        String model_choices
        Boolean use_dimensionality_reduction = false
        Boolean skip_ML_models = false
    }
    String pipeline_version = "1.0.0"
    String container_gen = "docker.io/library/proteomics:~{pipeline_version}"
    String container_vae = "docker.io/library/vae:~{pipeline_version}"
    if (use_dimensionality_reduction) {
        call pre.preprocessing_dim {
            input: 
                input_csv = input_csv,
                output_prefix = output_prefix,
                docker = container_gen
        }
    }
    if (!use_dimensionality_reduction) {
        call pre.preprocessing_std {
            input: 
                input_csv = input_csv,
                output_prefix = output_prefix,
                docker = container_gen
        }
    }
    File data_csv = select_first([preprocessing_dim.csv, preprocessing_std.csv])
    if (!skip_ML_models) {
        call cls.classification_gen {
            input:
                input_csv = data_csv,
                output_prefix = output_prefix,
                model = model_choices,
                docker = container_gen
        }
        call cls.classification_vae {
            input:
                input_csv = data_csv,
                output_prefix = output_prefix,
                model = model_choices,
                docker = container_vae
        }
        Array[File] cls_data_npy = flatten([classification_gen.data_npy, classification_vae.data_npy])
        Array[File] cls_model_pkl = flatten([classification_gen.model_pkl, classification_vae.model_pkl])
        Array[File] cls_data_pkl = flatten([classification_gen.data_pkl, classification_vae.data_pkl])
        Array[File] cls_metrics_plot = flatten([classification_gen.metrics_plot, classification_vae.metrics_plot])
        Array[File] cls_roc_curve_plot = flatten([classification_gen.roc_curve_plot, classification_vae.roc_curve_plot])
        Array[File] cls_confusion_matrix_plot = flatten([classification_gen.confusion_matrix_plot, classification_vae.confusion_matrix_plot])
        Array[File] vae_shap_out = flatten([classification_vae.vae_shap_csv])
        call report.plot {
            input:
                data_npy = cls_data_npy,
                model_pkl = cls_model_pkl,
                data_pkl = cls_data_pkl,
                vae_shap = vae_shap_out,
                model = model_choices,
                output_prefix = output_prefix,
                docker = container_gen
        }
        call report.pdf {
            input:
                confusion_matrix = cls_confusion_matrix_plot,
                roc_curve = cls_roc_curve_plot,
                joint_roc_curve = plot.all_roc_curves,
                metrics = cls_metrics_plot,
                vae_shap = vae_shap_out,
                model = model_choices,
                output_prefix = output_prefix,
                docker = container_gen
        }
    }
    output {
        File processed_csv = data_csv
        Array[File]? confusion_matrix_plot = cls_confusion_matrix_plot
        Array[File]? roc_curve_plot = cls_roc_curve_plot
        Array[File]? metrics_plot = cls_metrics_plot
        Array[File]? data_pkl = cls_data_pkl
        Array[File]? model_pkl = cls_model_pkl
        Array[File]? data_npy = cls_data_npy
        File? overall_roc_plot = plot.all_roc_curves
        Array[File]? shap_radar_plot = plot.radar_plot
        Array[File]? shap_values = plot.shap_values
        File? pdf_summary = pdf.report
    }
}