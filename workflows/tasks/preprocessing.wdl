version 1.0

task preprocessing {
    input {
        String docker 
    }

    command <<<
        set -euo pipefail
        python /scripts/Step1-Zscores.py \
            -i proDataLabel.csv \
            -p standardized_data
    >>>

    output {

    }

    runtime {
        docker: "~{docker}"
        memory: "16G"
        disks: "local-disk 40 HDD"
    }
}