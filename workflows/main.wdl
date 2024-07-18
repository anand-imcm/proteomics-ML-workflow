version 1.0

import "./tasks/preprocessing.wdl" as p

workflow main {
    input {
        File input_csv
        String output_prefix
    }
    String pipeline_version = "1.0.0"
    String container_src = "docker.io/library/proteomics:~{pipeline_version}"
    call p.preprocessing {
        input: 
            input_csv = input_csv,
            output_prefix = output_prefix,
            docker = container_src
    }
    output {
        File processed_csv = preprocessing.csv
    }
}