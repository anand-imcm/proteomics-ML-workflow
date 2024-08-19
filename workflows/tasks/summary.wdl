version 1.0

task plot {
    input {
        Array[File] data_npy
        Array[File] model_pkl
        Array[File] data_pkl
        Array[File] vae_shap
        String model
        String output_prefix
        String docker
        Int shap_radar_num_features = 10
        Int memory_gb = 24
        Int cpu = 16
    }
    Array[File] all_data = flatten([data_npy, model_pkl, data_pkl, vae_shap])
    Int disk_size_gb = ceil(size(all_data, "GB")) + 2
    command <<<
        set -euo pipefail
        for file_name in ~{sep=' ' all_data}; do
            cp $file_name $(basename $file_name)
        done
        python /scripts/Step3_OverallROC.py \
            -m ~{model} \
            -p ~{output_prefix}
        python /scripts/Step4_calculate_shap.py \
            -m ~{model} \
            -p ~{output_prefix} \
            -f ~{shap_radar_num_features}
    >>>
    output {
        File? all_roc_curves = output_prefix + "_overall_roc_curves.png"
        Array[File] radar_plot = glob("*_shap_radar.png")
        Array[File] shap_values = glob("*_shap_values.csv")
    }
    runtime {
        docker: "~{docker}"
        cpu: "~{cpu}"
        memory: "~{memory_gb}GB"
        disks: "local-disk ~{disk_size_gb} HDD"
    }
}

task dim {
    input {
        Array[File] dim_reduct_plot
        Array[File] dim_reduct_data
        String output_prefix
        String docker
        Int memory_gb = 24
        Int cpu = 16
    }
    Array[Array[File]] dim_results = [dim_reduct_plot, dim_reduct_data]
    Array[File] all_data = flatten(dim_results)
    Int disk_size_gb = ceil(size(all_data, "GB")) + 2
    command <<<
        set -euo pipefail
        for file_name in ~{sep=' ' all_data}; do
            cp $file_name $(basename $file_name)
        done
        python /scripts/Step5_PDF_summary_analysis.py \
            -p ~{output_prefix}
        mv "~{output_prefix}_report.pdf" "~{output_prefix}_dimensionality_reduction_report.pdf"
    >>>
    output {
        File report = output_prefix + "_dimensionality_reduction_report.pdf"
    }
    runtime {
        docker: "~{docker}"
        cpu: "~{cpu}"
        memory: "~{memory_gb}GB"
        disks: "local-disk ~{disk_size_gb} HDD"
    }
}

task summary {
    input {
        Array[File] summary_data
        String output_prefix
        String docker
        Int memory_gb = 24
        Int cpu = 16
    }
    Int disk_size_gb = ceil(size(summary_data, "GB")) + 2
    command <<<
        set -euo pipefail
        for file_name in ~{sep=' ' summary_data}; do
            cp $file_name $(basename $file_name)
        done
        python /scripts/Step5_PDF_summary_analysis.py \
            -p ~{output_prefix}
        mv "~{output_prefix}_report.pdf" "~{output_prefix}_analysis_report.pdf"
    >>>
    output {
        File report = output_prefix + "_analysis_report.pdf"
    }
    runtime {
        docker: "~{docker}"
        cpu: "~{cpu}"
        memory: "~{memory_gb}GB"
        disks: "local-disk ~{disk_size_gb} HDD"
    }
}