version 1.0

import "./tasks/run_plan.wdl" as RP

workflow main {
    input {
        File input_csv
        String output_prefix
        String? dimensionality_reduction_choices # t-SNE PCA
        String model_choices = "RF"
        String regression_choices = "NNR"
        String mode = "Classification" # choices: Classification, Regression, Summary
        Boolean calculate_shap = false
        Int shap_features = 20
        Int number_of_dimensions = 3
    }
    String pipeline_version = "1.0.2"
    String container_gen = "ghcr.io/anand-imcm/proteomics-ml-workflow-gen:~{pipeline_version}"
    Array[File] default_arr = []
    call RP.run_plan {
        input: model_choices = model_choices,
            dimensionality_reduction_choices = dimensionality_reduction_choices,
            regression_choices = regression_choices,
            mode = mode,
            shap = calculate_shap
    }
    if (run_plan.use_dim){
        scatter (dim_method in run_plan.dim_opt) {
            call dim_reduction {
                input:
                    input_csv = input_csv,
                    output_prefix = output_prefix,
                    dim_method = dim_method,
                    num_dimensions = number_of_dimensions,
                    docker = container_gen
            }
        }
    }
    Array[File] dim_out = if (run_plan.use_dim) then flatten(select_all([dim_reduction.out])) else default_arr
    Array[File] dim_png = if (run_plan.use_dim) then flatten(select_all([dim_reduction.png])) else default_arr
    if (run_plan.use_gen) {
        scatter (gen_method in run_plan.gen_opt) {
            call ml_gen {
                input:
                    model = gen_method,
                    input_csv = input_csv,
                    output_prefix = output_prefix,
                    dim_opt = run_plan.dim_opt[0],
                    docker = container_gen
            }
        }
    }
    if (run_plan.use_vae) {
        scatter (vae_method in run_plan.vae_opt) {
            call ml_vae {
                input:
                    model = vae_method,
                    input_csv = input_csv,
                    output_prefix = output_prefix,
                    dim_opt = run_plan.dim_opt[0],
                    docker = container_gen
            }
        }
    }
    Array[File] gen_ml_out = if (run_plan.use_gen) then flatten(select_all([ml_gen.data])) else default_arr
    Array[File] vae_ml_out = if (run_plan.use_vae) then flatten(select_all([ml_vae.data])) else default_arr
    Array[File] classification_out = flatten([gen_ml_out, vae_ml_out])
    if (run_plan.use_reg) {
        scatter (vae_method in run_plan.reg_opt) {
            call reg {
                input: model = vae_method, data = input_csv
            }
        }
    }
    Array[File] reg_out = if (run_plan.use_reg) then flatten(select_all([reg.out])) else default_arr
    Array[String] model_opts = flatten([run_plan.gen_opt, run_plan.vae_opt])
    Array[File] model_data = if (!run_plan.use_dim) then flatten([classification_out, reg_out]) else default_arr
    if (!run_plan.use_dim){
        call summary {
            input:
                dataset = model_data,
                model = model_opts,
                output_prefix = output_prefix,
                use_shap = run_plan.use_shap,
                shap_num_feat = shap_features,
                docker = container_gen
        }
    }
    Array[File] summary_files = if (!run_plan.use_dim) then select_all([summary.results]) else default_arr
    Array[File] all_results = flatten([dim_out, dim_png, summary_files])
    call pdf_report {
        input:
            summary_set = all_results,
            output_prefix = output_prefix,
            docker = container_gen
    }
    
    output {
        Array[String] dim_plan = run_plan.dim_opt
        Array[String] vae_plan = run_plan.vae_opt
        Array[String] gen_plan = run_plan.gen_opt
        Array[String] reg_plan = run_plan.reg_opt
        Boolean use_dim = run_plan.use_dim
        Boolean use_vae = run_plan.use_vae
        Boolean use_gen = run_plan.use_gen
        Boolean use_reg = run_plan.use_reg
        Boolean use_shap = run_plan.use_shap
        File report = pdf_report.out
        File results = pdf_report.results
    }
}

task dim_reduction {
    input {
        File input_csv
        String output_prefix
        String dim_method = "PCA"
        Int num_dimensions = 3
        String docker
        Int memory_gb = 16
        Int cpu = 16
    }
    Int disk_size_gb = ceil(size(input_csv, "GB")) + 5
    command <<<
        set -euo pipefail
        python /scripts/Step1_Preprocessing.py \
            -i ~{input_csv} \
            -m ~{dim_method} \
            -d ~{num_dimensions} \
            -p ~{output_prefix}
        touch ~{output_prefix}"_"~{dim_method}"_result.png"
    >>>
    output {
        File out = output_prefix + "_" + dim_method +"_result.csv"
        File png = output_prefix + "_" + dim_method +"_result.png"
    }
    runtime {
        docker: "~{docker}"
        cpu: "~{cpu}"
        memory: "~{memory_gb}GB"
        disks: "local-disk ~{disk_size_gb} HDD"
    }
}

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
        File confusion_matrix_plot = glob("*_confusion_matrix.png")[0]
        File data_pkl = glob("*_data.pkl")[0]
        File metrics_plot = glob("*_metrics.png")[0]
        File model_pkl = glob("*_model.pkl")[0]
        File out = glob("*_predictions.csv")[0]
        File roc_curve_plot = glob("*_roc_curve.png")[0]
        File roc_data = glob("*_roc_data.npy")[0]
        File data = output_prefix + "_" + model + "_results.tar.gz"
    }
    runtime {
        docker: "~{docker}"
        cpu: "~{cpu}"
        memory: "~{memory_gb}GB"
        disks: "local-disk ~{disk_size_gb} HDD"
    }
}

task ml_vae {
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
        File confusion_matrix_plot = glob("*_confusion_matrix.png")[0]
        File data_pkl = glob("*_data.pkl")[0]
        File metrics_plot = glob("*_metrics.png")[0]
        File model_pkl = glob("*_model.pkl")[0]
        File out = glob("*_predictions.csv")[0]
        File roc_curve_plot = glob("*_roc_curve.png")[0]
        File roc_data = glob("*_roc_data.npy")[0]
        File data = output_prefix + "_" + model + "_results.tar.gz"
    }
    runtime {
        docker: "~{docker}"
        cpu: "~{cpu}"
        memory: "~{memory_gb}GB"
        disks: "local-disk ~{disk_size_gb} HDD"
    }
}

task reg {
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

task summary {
    input {
        Array[File] dataset
        Array[String] model
        String output_prefix
        Boolean use_shap
        Int shap_num_feat = 20
        String docker
        Int memory_gb = 24
        Int cpu = 16
    }
    Array[File] all_data = flatten([dataset])
    Int disk_size_gb = ceil(size(all_data, "GB")) + 2
    command <<<
        set -euo pipefail
        for file_name in ~{sep=' ' all_data}; do
            cp $file_name $(basename $file_name)
        done
        if ls *.tar.gz 1> /dev/null 2>&1; then
            for archive in *.tar.gz; do
                tar -xzvf "$archive"
            done
        fi
        python /scripts/Classification/Step3_OverallROC.py \
            -m ~{sep=' ' model} \
            -p ~{output_prefix}
        if [ "~{use_shap}" = "true" ]; then
            python /scripts/Classification/Step4_Classification_SHAP.py \
                -p ~{output_prefix} \
                -m ~{sep=' ' model} \
                -f ~{shap_num_feat}
        fi
        tar -czvf ~{output_prefix}_results.tar.gz --ignore-failed-read *.{png,pkl,npy,csv}
    >>>
    output {
        File results = output_prefix + "_results.tar.gz"
    }
    runtime {
        docker: "~{docker}"
        cpu: "~{cpu}"
        memory: "~{memory_gb}GB"
        disks: "local-disk ~{disk_size_gb} HDD"
    }
}

task pdf_report {
    input {
        Array[File] summary_set
        String output_prefix
        String docker
        Int memory_gb = 24
        Int cpu = 16
    }
    Array[File] all_data = flatten([summary_set])
    Int disk_size_gb = ceil(size(all_data, "GB")) + 2
    command <<<
        set -euo pipefail
        for file_name in ~{sep=' ' all_data}; do
            cp $file_name $(basename $file_name)
        done
        if ls *.tar.gz 1> /dev/null 2>&1; then
            for archive in *.tar.gz; do
                tar -xzvf "$archive"
            done
            rm *.tar.gz
        fi
        if [ -f ~{output_prefix}_elasticnet_result.png ]; then
            rm ~{output_prefix}_elasticnet_result.png
        fi
        python /scripts/Step5_PDF_summary_analysis.py \
            -p ~{output_prefix}
        tar -czvf ~{output_prefix}_results.tar.gz --ignore-failed-read *.{png,pkl,npy,csv}
    >>>
    output {
        File out = "model_reports.pdf"
        File results = output_prefix + "_results.tar.gz"
    }
    runtime {
        docker: "~{docker}"
        cpu: "~{cpu}"
        memory: "~{memory_gb}GB"
        disks: "local-disk ~{disk_size_gb} HDD"
    }
}