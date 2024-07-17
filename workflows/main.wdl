version 1.0

import "./tasks/preprocessing.wdl" as p

workflow main {
    String pipeline_version = "1.0.0"
    String container_src = "docker.io/library/proteomics"

    call p.preprocessing {
        input: docker = container_src
    }
}