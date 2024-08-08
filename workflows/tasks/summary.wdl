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
        Int memory_gb = 16
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
        File all_roc_curves = output_prefix + "_overall_roc_curves.png"
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

task pdf {
    input {
        Array[File] confusion_matrix
        Array[File] roc_curve
        File joint_roc_curve
        Array[File] metrics
        Array[File] vae_shap
        String model
        String output_prefix
        String docker
        Int memory_gb = 16
        Int cpu = 16
    }
    Array[File] all_data = flatten([confusion_matrix, roc_curve, metrics, vae_shap])
    Int disk_size_gb = ceil(size(all_data, "GB")) + 2
    
    command <<<
        set -euo pipefail
        cp ~{joint_roc_curve} .
        for file_name in ~{sep=' ' all_data}; do
            cp $file_name $(basename $file_name)
        done
        python /scripts/Step5_PDF_summary.py \
            -m ~{model} \
            -p ~{output_prefix}
    >>>
    output {
        File report = output_prefix + "_model_reports.pdf"
    }
    runtime {
        docker: "~{docker}"
        cpu: "~{cpu}"
        memory: "~{memory_gb}GB"
        disks: "local-disk ~{disk_size_gb} HDD"
    }
}