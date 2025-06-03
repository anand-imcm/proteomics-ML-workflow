version 1.0

task ppi_analysis {
    input {
        File? summary_set
        File proteinExpFile
        String output_prefix
        Int score_threshold = 400
        Int combined_score_threshold = 800
        Int SHAP_threshold = 100
        Boolean protein_name_mapping = true
        String correlation_method = "spearman"
        Float correlation_threshold = 0.8
        String docker = "ghcr.io/anand-imcm/proteomics-ml-workflow-net:1.0.2"
        Int memory_gb = 24
        Int cpu = 16
    }
    command <<<
        set -euo pipefail
        cp ~{summary_set} $(basename ~{summary_set})
        if ls *.tar.gz 1> /dev/null 2>&1; then
            for archive in *.tar.gz; do
                tar -xzvf "$archive"
            done
            rm *.tar.gz
        fi
        patternChosen="shap_values.csv"
        converProId=""
        if ~{protein_name_mapping}; then
            converProId="TRUE"
        else
            converProId="FALSE"
        fi
        Rscript /scripts/protein_network.R \
            --score_thresholdHere ~{score_threshold} \
            --combined_score_thresholdHere ~{combined_score_threshold} \
            --SHAPthresh ~{SHAP_threshold} \
            --patternChosen ${patternChosen} \
            --converProId ${converProId} \
            --proteinExpFile ~{proteinExpFile} \
            --CorMethod ~{correlation_method} \
            --CorThreshold ~{correlation_threshold}
        
        tar -czvf ~{output_prefix}_pro_net_results.tar.gz --ignore-failed-read Network*.png *.xlsx
    >>>
    output {
        File results = output_prefix + "_pro_net_results.tar.gz"
    }
    runtime {
        docker: "~{docker}"
        cpu: "~{cpu}"
        memory: "~{memory_gb}GB"
        disks: "local-disk 20 HDD"
        continueOnReturnCode: [0, 1]
    }
}