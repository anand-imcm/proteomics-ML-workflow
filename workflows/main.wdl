version 1.0

import "./tasks/preprocessing.wdl" as p
import "./tasks/classification.wdl" as ml


workflow main {
    input {
        File input_csv
        String output_prefix
        String model
    }
    String pipeline_version = "1.0.0"
    String container_src = "docker.io/library/proteomics:~{pipeline_version}"
    call p.preprocessing {
        input: 
            input_csv = input_csv,
            output_prefix = output_prefix,
            docker = container_src
    }
    call ml.classification {
        input:
            input_csv = preprocessing.csv,
            output_prefix = output_prefix,
            model = model,
            docker = container_src
    }
    output {
        File processed_csv = preprocessing.csv
        Array[File] confusion_matrix_plot = classification.confusion_matrix_plot
        Array[File] roc_curve_plot = classification.roc_curve_plot
        Array[File] metrics_plot = classification.metrics_plot
        Array[File] data_pkl = classification.data_pkl
        Array[File] model_pkl = classification.model_pkl
        Array[File] data_npy = classification.data_npy
    }
}