version 1.0

import "./tasks/classification.wdl" as ml
import "./tasks/summary.wdl" as std_ml_report

workflow standard_ml_wf {
    input {
        File input_csv
        String output_prefix
        String container_gen
        String container_vae
        String model_choices
    }
    call ml.classification_gen as gen {
        input:
            input_csv = input_csv,
            output_prefix = output_prefix,
            model = model_choices,
            docker = container_gen
    }
    call ml.classification_vae as vae {
        input:
            input_csv = input_csv,
            output_prefix = output_prefix,
            model = model_choices,
            docker = container_vae
    }
    Array[File] cls_data_npy = flatten([gen.data_npy, vae.data_npy])
    Array[File] cls_model_pkl = flatten([gen.model_pkl, vae.model_pkl])
    Array[File] cls_data_pkl = flatten([gen.data_pkl, vae.data_pkl])
    Array[File] cls_metrics_plot = flatten([gen.metrics_plot, vae.metrics_plot])
    Array[File] cls_roc_curve_plot = flatten([gen.roc_curve_plot, vae.roc_curve_plot])
    Array[File] cls_confusion_matrix_plot = flatten([gen.confusion_matrix_plot, vae.confusion_matrix_plot])
    Array[File] vae_shap_out = flatten([vae.vae_shap_csv])
    call std_ml_report.plot as roc_shap_summary {
        input:
            data_npy = cls_data_npy,
            model_pkl = cls_model_pkl,
            data_pkl = cls_data_pkl,
            vae_shap = vae_shap_out,
            model = model_choices,
            output_prefix = output_prefix,
            docker = container_gen
    }
    output {
        Array[File] out_cls_data_npy = cls_data_npy
        Array[File] out_cls_model_pkl = cls_model_pkl
        Array[File] out_cls_data_pkl = cls_data_pkl
        Array[File] out_cls_metrics_plot = cls_metrics_plot
        Array[File] out_cls_roc_curve_plot = cls_roc_curve_plot
        Array[File] out_cls_confusion_matrix_plot = cls_confusion_matrix_plot
        Array[File] out_vae_shap_out = vae_shap_out
        File? out_all_roc_curves = roc_shap_summary.all_roc_curves
        Array[File] out_radar_plot = roc_shap_summary.radar_plot
        Array[File] out_shap_values = roc_shap_summary.shap_values
    }
}