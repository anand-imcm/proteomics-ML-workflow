version 1.0

task roc_plot {
    input {
        Array[File] data_npy
        String model
        String output_prefix
        String docker
        Int memory_gb = 8
        Int cpu = 8
    }
    Int disk_size_gb = ceil(size(data_npy, "GB")) + 2
    
    command <<<
        set -euo pipefail
        for npy in ~{sep=' ' data_npy}; do
            ln -s $npy $(basename $npy)
        done
        python /scripts/Step3_OverallROC.py \
            -m ~{model} \
            -p ~{output_prefix}
    >>>
    output {
        File png = output_prefix + "_overall_roc_curves.png"
    }
    runtime {
        docker: "~{docker}"
        cpu: "~{cpu}"
        memory: "~{memory_gb}GB"
        disks: "local-disk ~{disk_size_gb} HDD"
    }
}