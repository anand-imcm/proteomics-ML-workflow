version 1.0

task run_plan {
    input {
        String model_choices
        String regression_choices
        String? dimensionality_reduction_choices
        String mode
        Boolean shap
        Boolean ppi
    }
    command <<<
        python3 <<EOF
        import re
        usr_models = "~{model_choices}"
        usr_dim = "~{dimensionality_reduction_choices}"
        usr_reg = "~{regression_choices}"
        run_mode = "~{mode}"
        usr_shap = "~{shap}"
        usr_ppi = "~{ppi}"
        dim_options = re.split(r'\s+|,', usr_dim)
        ml_options = re.split(r'\s+|,', usr_models)
        reg_options = re.split(r'\s+|,', usr_reg)
        vae_choices = [opt for opt in ml_options if "VAE" in opt]
        gen_choices = [opt for opt in ml_options if "VAE" not in opt]
        ppi_compatibility = False
        with open("use_gen.txt", "w") as gen_opt, open("use_reg.txt", "w") as reg_opt, open("use_vae.txt", "w") as vae_opt, open("use_shap.txt", "w") as shap_opt,  open("use_dim.txt", "w") as dim_opt, open("use_ppi.txt", "w") as ppi_opt, open("dim_options.txt", "w") as dim_plan, open("cl_options.txt", "w") as cl_plan, open("vae_options.txt", "w") as vae_plan, open("reg_options.txt", "w") as reg_plan:
            dim_opt.write("false")
            gen_opt.write("false")
            reg_opt.write("false")
            vae_opt.write("false")
            shap_opt.write("false")
            ppi_opt.write("false")
            if run_mode.lower() == "regression":
                if any(dim_options) and len(dim_options) > 1:
                    dim_opt.seek(0)
                    dim_opt.truncate()
                    dim_opt.write("true")
                    for dim in dim_options:
                        dim_plan.write(dim.replace("-", "").lower() + "\n")
                else:
                    if any(dim_options):
                        dim_plan.write(dim_options[0].replace("-", "").lower())
                        if dim_options[0].replace("-", "").lower() == "elasticnet":
                            ppi_compatibility = True
                    else:
                        ppi_compatibility = True
                        dim_plan.write("none")
                    if any(reg_options):
                        reg_opt.seek(0)
                        reg_opt.truncate()
                        reg_opt.write("true")
                        for reg in reg_options:
                            reg_plan.write(reg + "\n")
                    if usr_shap.lower() == "true":
                        shap_opt.seek(0)
                        shap_opt.truncate()
                        shap_opt.write("true")
                    if usr_ppi.lower() == "true" and ppi_compatibility:
                        shap_opt.seek(0)
                        shap_opt.truncate()
                        shap_opt.write("true")
                        ppi_opt.seek(0)
                        ppi_opt.truncate()
                        ppi_opt.write("true")
            elif run_mode.lower() == "classification":
                if any(dim_options) and len(dim_options) > 1:
                    dim_opt.seek(0)
                    dim_opt.truncate()
                    dim_opt.write("true")
                    for dim in dim_options:
                        dim_plan.write(dim.replace("-", "").lower() + "\n")
                else:
                    if any(dim_options):
                        dim_plan.write(dim_options[0].replace("-", "").lower())
                        if dim_options[0].replace("-", "").lower() == "elasticnet":
                            ppi_compatibility = True
                    else:
                        dim_plan.write("none")
                        ppi_compatibility = True
                    if any(gen_choices):
                        gen_opt.seek(0)
                        gen_opt.truncate()
                        gen_opt.write("true")
                        for ml in gen_choices:
                            cl_plan.write(ml + "\n")
                    if any(vae_choices):
                        vae_opt.seek(0)
                        vae_opt.truncate()
                        vae_opt.write("true")
                        for ml in vae_choices:
                            vae_plan.write(ml + "\n")
                    if usr_shap.lower() == "true":
                        shap_opt.seek(0)
                        shap_opt.truncate()
                        shap_opt.write("true")
                    if usr_ppi.lower() == "true" and ppi_compatibility:
                        shap_opt.seek(0)
                        shap_opt.truncate()
                        shap_opt.write("true")
                        ppi_opt.seek(0)
                        ppi_opt.truncate()
                        ppi_opt.write("true")
            else:
                run_mode = "summary"
                if any(dim_options):
                    dim_opt.seek(0)
                    dim_opt.truncate()
                    dim_opt.write("true")
                    for dim in dim_options:
                        dim_plan.write(dim.replace("-", "").lower() + "\n")
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
        Boolean use_ppi = read_boolean("use_ppi.txt")
    }
    runtime {
        docker: "python:3.12-slim"
        cpu: "2"
        memory: "2GB"
        disks: "local-disk 2 HDD"
    }
}