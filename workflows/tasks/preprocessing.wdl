version 1.0

task preprocessing_std {
    input {
        File input_csv
        String output_prefix 
        String docker
        Int memory_gb = 8
        Int cpu = 8
    }
    Int disk_size_gb = ceil(size(input_csv, "GB")) + 2
    command <<<
        set -euo pipefail
        python /scripts/Step1_Zscores.py \
            -i ~{input_csv} \
            -p ~{output_prefix}
    >>>
    output {
        File csv = output_prefix + ".csv"
    }
    runtime {
        docker: "~{docker}"
        cpu: "~{cpu}"
        memory: "~{memory_gb}GB"
        disks: "local-disk ~{disk_size_gb} HDD"
    }
}