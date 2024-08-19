version 1.0

import "./tasks/preprocessing.wdl" as dim
import "./tasks/summary.wdl" as dim_report

workflow dim_reduction_wf {
    input {
        File input_csv
        String output_prefix
        String docker
        String method_name = "PCA"
        Int num_of_dimensions = 3
    }
    call dim.preprocessing_dim as dim_reduction_wf {
        input: 
            input_csv = input_csv,
            output_prefix = output_prefix,
            docker = docker,
            dim_reduction_method = method_name,
            num_dimensions = num_of_dimensions
    }
    call dim_report.dim as summary_wf {
        input:
            dim_reduct_plot = dim_reduction_wf.png_list,
            dim_reduct_data = dim_reduction_wf.csv_list,
            output_prefix = output_prefix,
            docker = docker
    }
    output {
        Array[File] csv_list = dim_reduction_wf.csv_list
        Array[File] png_list = dim_reduction_wf.png_list
        File csv = dim_reduction_wf.csv
        File png = dim_reduction_wf.png
        File report = summary_wf.report
    }
}