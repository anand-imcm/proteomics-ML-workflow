version 1.0

task ml_reg {
    input {
        String model
        File data
        String docker
        Int memory_gb = 24
        Int cpu = 16
    }
    Int disk_size_gb = ceil(size(data, "GB")) + 5
    command <<<
        set -euo pipefail
        wc -l ~{data}
        echo "running REG task with ~{model}" > ~{model}.txt
    >>>
    output {
        File out = "~{model}.txt"
    }
    runtime {
        docker: "~{docker}"
        cpu: "~{cpu}"
        memory: "~{memory_gb}GB"
        disks: "local-disk ~{disk_size_gb} HDD"
    }
}