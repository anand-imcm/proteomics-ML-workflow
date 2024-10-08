version 1.0

task classification_gen {
    
    input {
        File input_csv
        String model
        String output_prefix 
        String docker
        Int memory_gb = 24
        Int cpu = 16
    }
    
    Int disk_size_gb = ceil(size(input_csv, "GB")) + 5
    
    command <<<
        set -euo pipefail
        filename=~{input_csv}
        if [[ $filename == *.tar.gz ]]; then
            tar -xzf "$filename"
            filename=$(find . -maxdepth 1 -type f -name "*.csv" | head -n 1)
        fi
        printf "Running classification using $filename\n"
        model_modified=$(echo "~{model}" | sed 's/VAE//g')
        python /scripts/classification.py \
            -i $filename \
            -p ~{output_prefix} \
            -m ${model_modified}
    >>>
    
    output {
        Array[File] confusion_matrix_plot = glob("*_confusion_matrix.png")
        Array[File] roc_curve_plot = glob("*_roc_curve.png")
        Array[File] metrics_plot = glob("*_metrics.png")
        Array[File] data_pkl = glob("*_data.pkl")
        Array[File] model_pkl = glob("*_model.pkl")
        Array[File] data_npy = glob("*_data.npy")
    }
    
    runtime {
        docker: "~{docker}"
        cpu: "~{cpu}"
        memory: "~{memory_gb}GB"
        disks: "local-disk ~{disk_size_gb} HDD"
        continueOnReturnCode: [0,2]
    }
}

task classification_vae {
    
    input {
        File input_csv
        String model
        String output_prefix 
        String docker
        Int memory_gb = 24
        Int cpu = 16
    }
    
    Int disk_size_gb = ceil(size(input_csv, "GB")) + 5
    
    command <<<
        set -euo pipefail
        filename=~{input_csv}
        if [[ $filename == *.tar.gz ]]; then
            tar -xzf "$filename"
            filename=$(find . -maxdepth 1 -type f -name "*.csv" | head -n 1)
        fi
        printf "Running classification using $filename\n"
        if [[ "~{model}" == *"VAE"* ]]; then
            python /scripts/classification.py \
            -i $filename \
            -p ~{output_prefix} \
            -m VAE
        fi
    >>>
    
    output {
        Array[File] confusion_matrix_plot = glob("*_confusion_matrix.png")
        Array[File] roc_curve_plot = glob("*_roc_curve.png")
        Array[File] metrics_plot = glob("*_metrics.png")
        Array[File] data_pkl = glob("*_data.pkl")
        Array[File] model_pkl = glob("*_model.pkl")
        Array[File] data_npy = glob("*_data.npy")
        Array[File] vae_shap_csv = glob("*_shap_values.csv")
    }
    
    runtime {
        docker: "~{docker}"
        cpu: "~{cpu}"
        memory: "~{memory_gb}GB"
        disks: "local-disk ~{disk_size_gb} HDD"
    }
}