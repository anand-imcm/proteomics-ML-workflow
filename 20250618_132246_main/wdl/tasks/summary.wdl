version 1.0

task summary {
    input {
        Array[File] dataset
        Array[String] model
        String output_prefix
        Boolean use_shap
        Boolean use_reg
        Int shap_num_feat = 20
        String docker
        Int memory_gb = 24
        Int cpu = 16
    }
    Array[File] all_data = flatten([dataset])
    Int disk_size_gb = ceil(size(all_data, "GB")) + 2
    command <<<
        set -euo pipefail
        for file_name in ~{sep=' ' all_data}; do
            cp $file_name $(basename $file_name)
        done
        if ls *.tar.gz 1> /dev/null 2>&1; then
            for archive in *.tar.gz; do
                tar -xzvf "$archive"
            done
        fi
        if [ "~{use_reg}" = "false" ]; then
            python /scripts/Classification/Step3_OverallROC.py \
                -m ~{sep=' ' model} \
                -p ~{output_prefix}
        fi
        if [ "~{use_shap}" = "true" ]; then
            if [ "~{use_reg}" = "true" ]; then
                python /scripts/Regression/Step3_Regression_SHAP.py \
                    --p ~{output_prefix} \
                    --m ~{sep=' ' model} \
                    --f ~{shap_num_feat}
            else
                python /scripts/Classification/Step4_Classification_SHAP.py \
                    -p ~{output_prefix} \
                    -m ~{sep=' ' model} \
                    -f ~{shap_num_feat}
            fi
        fi
        tar -czvf ~{output_prefix}_results.tar.gz --ignore-failed-read *.{png,pkl,npy,csv}
    >>>
    output {
        File results = output_prefix + "_results.tar.gz"
    }
    runtime {
        docker: "~{docker}"
        cpu: "~{cpu}"
        memory: "~{memory_gb}GB"
        disks: "local-disk ~{disk_size_gb} HDD"
    }
}