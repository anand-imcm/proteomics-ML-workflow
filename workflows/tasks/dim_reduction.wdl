version 1.0

task dim_reduction {
    input {
        File input_csv
        String output_prefix
        String dim_method = "PCA"
        Int num_dimensions = 3
        String docker
        Int memory_gb = 16
        Int cpu = 16
    }
    Int disk_size_gb = ceil(size(input_csv, "GB")) + 5
    command <<<
        set -euo pipefail
        python /scripts/Step1_Preprocessing.py \
            -i ~{input_csv} \
            -m ~{dim_method} \
            -d ~{num_dimensions} \
            -p ~{output_prefix}
        touch ~{output_prefix}"_"~{dim_method}"_result.png"
    >>>
    output {
        File out = output_prefix + "_" + dim_method +"_result.csv"
        File png = output_prefix + "_" + dim_method +"_result.png"
    }
    runtime {
        docker: "~{docker}"
        cpu: "~{cpu}"
        memory: "~{memory_gb}GB"
        disks: "local-disk ~{disk_size_gb} HDD"
    }
}