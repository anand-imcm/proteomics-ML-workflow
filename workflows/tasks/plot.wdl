version 1.0

task plot {
    input {
        Array[File] data_npy
        Array[File] model_pkl
        Array[File] data_pkl
        String model
        String output_prefix
        String docker
        Int shap_radar_num_features = 3
        Int memory_gb = 16
        Int cpu = 16
    }
    Array[File] all_data = flatten([data_npy, model_pkl, data_pkl])
    Int disk_size_gb = ceil(size(all_data, "GB")) + 2
    
    command <<<
        set -euo pipefail
        for model in ~{sep=' ' all_data}; do
            ln -s $model $(basename $model)
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