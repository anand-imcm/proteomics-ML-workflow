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
        Int memory_gb_preprocessing = 64
        Int cpu_preprocessing = 32
        Int memory_gb_SHAP_summary = 64
        Int cpu_SHAP_summary = 32
    }
    
    call dim.preprocessing_dim as dim_reduction_wf {
        input: 
            input_csv = input_csv,
            output_prefix = output_prefix,
            docker = docker,
            dim_method = method_name,
            num_dimensions = num_of_dimensions,
            memory_gb = memory_gb_preprocessing,
            cpu = cpu_preprocessing
    }
    
    call dim_report.dim as summary_wf {
        input:
            dim_reduct_plot = dim_reduction_wf.png_list,
            dim_reduct_data = dim_reduction_wf.csv_list,
            output_prefix = output_prefix,
            docker = docker,
            memory_gb = memory_gb_SHAP_summary,
            cpu = cpu_SHAP_summary
    }
    
    output {
        Array[File] csv_list = dim_reduction_wf.csv_list
        Array[File] png_list = dim_reduction_wf.png_list
        File csv = dim_reduction_wf.csv
        File png = dim_reduction_wf.png
        File report = summary_wf.report
    }
}