version 1.0

import "./tasks/preprocessing.wdl" as p
import "./tasks/classification.wdl" as cls
import "./tasks/roc_plot.wdl" as plt

workflow main {
    input {
        File input_csv
        String output_prefix
        String model_choices
    }
    String pipeline_version = "1.0.0"
    String container_gen = "docker.io/library/proteomics:~{pipeline_version}"
    String container_vae = "docker.io/library/vae:~{pipeline_version}"

    call p.preprocessing_std {
        input: 
            input_csv = input_csv,
            output_prefix = output_prefix,
            docker = container_gen
    }
    call cls.classification_gen {
        input:
            input_csv = preprocessing_std.csv,
            output_prefix = output_prefix,
            model = model_choices,
            docker = container_gen
    }
    call cls.classification_vae {
        input:
            input_csv = preprocessing_std.csv,
            output_prefix = output_prefix,
            model = model_choices,
            docker = container_vae
    }
    Array[File] all_data_npy = flatten([classification_gen.data_npy, classification_vae.data_npy])
    Array[File] all_model_pkl = flatten([classification_gen.model_pkl, classification_vae.model_pkl])
    Array[File] all_data_pkl = flatten([classification_gen.data_pkl, classification_vae.data_pkl])
    Array[File] all_metrics_plot = flatten([classification_gen.metrics_plot, classification_vae.metrics_plot])
    Array[File] all_roc_curve_plot = flatten([classification_gen.roc_curve_plot, classification_vae.roc_curve_plot])
    Array[File] all_confusion_matrix_plot = flatten([classification_gen.confusion_matrix_plot, classification_vae.confusion_matrix_plot])

    call plt.roc_plot {
        input:
            data_npy = all_data_npy,
            model = model_choices,
            output_prefix = output_prefix,
            docker = container_gen
    }
    output {
        File processed_csv = preprocessing_std.csv
        Array[File] confusion_matrix_plot = all_confusion_matrix_plot
        Array[File] roc_curve_plot = all_roc_curve_plot
        Array[File] metrics_plot = all_metrics_plot
        Array[File] data_pkl = all_data_pkl
        Array[File] model_pkl = all_model_pkl
        Array[File] data_npy = all_data_npy
        File overall_roc_plot = roc_plot.png
    }
}