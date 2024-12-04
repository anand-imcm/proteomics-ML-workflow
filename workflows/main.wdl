version 1.0

workflow main {
    input {
        File input_csv
        String output_prefix
        String? dimensionality_reduction_choices
        String model_choices = "RF"
        String regression_choices = "NNR"
        String mode = "Classification" # choices: Classification, Regression, Summary
        Boolean calculate_shap = false
    }
    String pipeline_version = "1.0.1"
    String container_gen = "ghcr.io/anand-imcm/proteomics-ml-workflow-gen:~{pipeline_version}"
    String container_vae = "ghcr.io/anand-imcm/proteomics-ml-workflow-vae:~{pipeline_version}"
    Array[File] default_arr = []
    call run_plan {
        input: model_choices = model_choices,
            dimensionality_reduction_choices = dimensionality_reduction_choices,
            regression_choices = regression_choices,
            mode = mode,
            shap = calculate_shap
    }
    if (!run_plan.use_dim){
        call std_preprocessing {
            input: 
                input_csv = input_csv,
                output_prefix = output_prefix,
                docker = container_gen,
        }
    }
    if (run_plan.use_dim){
        scatter (dim_method in run_plan.dim_opt) {
            call dim_reduction {
                input:
                    input_csv = input_csv,
                    output_prefix = output_prefix,
                    dim_method = dim_method,
                    docker = container_gen
            }
        }
    }
    Array[File] std_out = if (!run_plan.use_dim) then flatten(select_all([std_preprocessing.out])) else default_arr
    Array[File] dim_out = if (run_plan.use_dim) then flatten(select_all([dim_reduction.out])) else default_arr
    Array[File] dim_png = if (run_plan.use_dim) then flatten(select_all([dim_reduction.png])) else default_arr
    Array[File] processed_csv = flatten([std_out, dim_out])
    if (run_plan.use_gen) {
        scatter (gen_method in run_plan.gen_opt) {
            call ml_gen {
                input:
                    model = gen_method,
                    input_csv = processed_csv[0],
                    output_prefix = output_prefix,
                    docker = container_gen
            }
        }
    }
    if (run_plan.use_vae) {
        scatter (vae_method in run_plan.vae_opt) {
            call ml_vae {
                input:
                    model = vae_method,
                    input_csv = processed_csv[0],
                    output_prefix = output_prefix,
                    docker = container_gen
            }
        }
    }
    Array[File] gen_ml_out = if (run_plan.use_gen) then flatten(select_all([ml_gen.out])) else default_arr
    Array[File] vae_ml_out = if (run_plan.use_vae) then flatten(select_all([ml_vae.out])) else default_arr
    Array[File] classification_out = flatten([gen_ml_out, vae_ml_out])
    if (run_plan.use_reg) {
        scatter (vae_method in run_plan.reg_opt) {
            call reg {
                input: model = vae_method, data = processed_csv[0]
            }
        }
    }
    Array[File] reg_out = if (run_plan.use_reg) then flatten(select_all([reg.out])) else default_arr

    Array[File] all_results = flatten([dim_out, dim_png, classification_out, reg_out])

    call pdf_report {
        input:
            summary_set = all_results,
            output_prefix = output_prefix,
            docker = container_gen
    }

    output {
        Array[File] std_preprocessing_out = std_out
        Array[File] dim_reduction_out = dim_out
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
        Array[File] ml_classification_out = classification_out
        Array[File] regression_out = reg_out
    }
}

task run_plan {
    input {
        String model_choices
        String regression_choices
        String? dimensionality_reduction_choices
        String mode
        Boolean shap
    }
    command <<<
        python3 <<EOF
        import re
        usr_models = "~{model_choices}"
        usr_dim = "~{dimensionality_reduction_choices}"
        usr_reg = "~{regression_choices}"
        run_mode = "~{mode}"
        usr_shap = "~{shap}"
        dim_options = re.split(r'\s+|,', usr_dim)
        ml_options = re.split(r'\s+|,', usr_models)
        reg_options = re.split(r'\s+|,', usr_reg)
        vae_choices = [opt for opt in ml_options if "VAE" in opt]
        gen_choices = [opt for opt in ml_options if "VAE" not in opt]
        with open("use_gen.txt", "w") as gen_opt, open("use_reg.txt", "w") as reg_opt, open("use_vae.txt", "w") as vae_opt, open("use_shap.txt", "w") as shap_opt,  open("use_dim.txt", "w") as dim_opt, open("dim_options.txt", "w") as dim_plan, open("cl_options.txt", "w") as cl_plan, open("vae_options.txt", "w") as vae_plan, open("reg_options.txt", "w") as reg_plan:
            if run_mode.lower() != "regression" and run_mode.lower() != "classification":
                run_mode = "summary"
                if any(dim_options):
                    dim_opt.write("true")
                    for dim in dim_options:
                        dim_plan.write(dim + "\n")
                else:
                    dim_opt.write("false")
                gen_opt.write("false")
                vae_opt.write("false")
                reg_opt.write("false")
                shap_opt.write("false")
            if run_mode.lower() == "regression":
                run_mode = "regression"
                if any(dim_options):
                    dim_opt.write("true")
                    dim_plan.write(dim_options[0])
                else:
                    dim_opt.write("false")
                gen_opt.write("false")
                vae_opt.write("false")
                if any(reg_options):
                    reg_opt.write("true")
                    for reg in reg_options:
                        reg_plan.write(reg + "\n")
                    if usr_shap.lower() == "true":
                        shap_opt.write("true")
                else:
                    reg_opt.write("false")
                    shap_opt.write("false")
            if run_mode.lower() == "classification":
                run_mode = "classification"
                shap_opt.write("false")
                reg_opt.write("false")
                if any(dim_options):
                    dim_opt.write("true")
                    dim_plan.write(dim_options[0])
                else:
                    dim_opt.write("false")
                if any(gen_choices):
                    gen_opt.write("true")
                    for ml in gen_choices:
                        cl_plan.write(ml + "\n")
                    if usr_shap.lower() == "true":
                        shap_opt.seek(0)
                        shap_opt.truncate()
                        shap_opt.write("true")
                else:
                    gen_opt.write("false")
                if any(vae_choices):
                    vae_opt.write("true")
                    for ml in vae_choices:
                        vae_plan.write(ml + "\n")
                    if usr_shap.lower() == "true":
                        shap_opt.seek(0)
                        shap_opt.truncate()
                        shap_opt.write("true")
                else:
                    vae_opt.write("false")
        EOF
    >>>
    output {
        Boolean use_dim = read_boolean("use_dim.txt")
        Array[String] dim_opt = read_lines("dim_options.txt")
        Boolean use_gen = read_boolean("use_gen.txt")
        Array[String] gen_opt = read_lines("cl_options.txt")
        Boolean use_vae = read_boolean("use_vae.txt")
        Array[String] vae_opt = read_lines("vae_options.txt")
        Boolean use_reg = read_boolean("use_reg.txt")
        Array[String] reg_opt = read_lines("reg_options.txt")
        Boolean use_shap = read_boolean("use_shap.txt")
    }
}

task std_preprocessing {
    input {
        File input_csv
        String output_prefix
        String docker
        Int memory_gb = 8
        Int cpu = 8
        
    }
    Int disk_size_gb = ceil(size(input_csv, "GB")) + 2
    command <<<
        set -euo pipefail
        python /scripts/Step1_Zscores.py \
            -i ~{input_csv} \
            -p ~{output_prefix}
    >>>
    output {
        Array[File] out = glob("*.csv")
        # File out = output_prefix + ".csv"
    }
    runtime {
        docker: "~{docker}"
        cpu: "~{cpu}"
        memory: "~{memory_gb}GB"
        disks: "local-disk ~{disk_size_gb} HDD"
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
        String docker
        Int memory_gb = 24
        Int cpu = 16
    }
    Int disk_size_gb = ceil(size(input_csv, "GB")) + 5
    command <<<
        set -euo pipefail
        echo "running ML task with ~{model}" > ~{model}.txt
        python /scripts/classification.py \
            -i ~{input_csv} \
            -p ~{output_prefix} \
            -m ~{model}
    >>>
    output {
        # File plot = "*_confusion_matrix.png"
        # File out = "*_predictions.csv"
        File out = glob("*_predictions.csv")[0]
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
        String docker
        Int memory_gb = 24
        Int cpu = 16
    }
    Int disk_size_gb = ceil(size(input_csv, "GB")) + 5
    command <<<
        set -euo pipefail
        echo "running ML task with ~{model}" > ~{model}.txt
        python /scripts/classification.py \
            -i ~{input_csv} \
            -p ~{output_prefix} \
            -m ~{model}
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

task reg {
    input {
        String model
        File data
    }
    command <<<
        wc -l ~{data}
        echo "running REG task with ~{model}" > ~{model}.txt
    >>>
    output {
        File out = "~{model}.txt"
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
        python /scripts/Step5_PDF_summary_analysis.py \
            -p ~{output_prefix}
    >>>
    output {
        File out = "model_reports.pdf"
    }
    runtime {
        docker: "~{docker}"
        cpu: "~{cpu}"
        memory: "~{memory_gb}GB"
        disks: "local-disk ~{disk_size_gb} HDD"
    }
}