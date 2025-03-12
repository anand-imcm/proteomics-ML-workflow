version 1.0

task ml_reg {
    input {
        String model
        File data
    }
    command <<<
        set -euo pipefail
        wc -l ~{data}
        echo "running REG task with ~{model}" > ~{model}.txt
    >>>
    output {
        File out = "~{model}.txt"
    }
}