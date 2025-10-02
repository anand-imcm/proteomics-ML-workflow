version 1.0

task ml_gen {
    input {
        String model
        File input_csv
        String output_prefix
        String dim_opt
        String docker
        Int memory_gb = 24
        Int cpu = 16
    }
    Int disk_size_gb = ceil(size(input_csv, "GB")) + 5
    command <<<
        set -euo pipefail
        python /scripts/Classification/classification.py \
            -i ~{input_csv} \
            -p ~{output_prefix} \
            -m ~{model} \
            -f ~{dim_opt}
        tar -czvf ~{output_prefix}_~{model}_results.tar.gz --ignore-failed-read *.{png,pkl,npy,csv}
    >>>
    output {
        File data = output_prefix + "_" + model + "_results.tar.gz"
    }
    runtime {
        docker: "~{docker}"
        cpu: "~{cpu}"
        memory: "~{memory_gb}GB"
        disks: "local-disk ~{disk_size_gb} HDD"
    }
}