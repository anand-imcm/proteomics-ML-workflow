version 1.0

task protein_network {
    input {
        File summary_set
        File proteinExpFile
        String output_prefix
        Int score_thresholdHere = 400
        Int combined_score_thresholdHere = 800
        Int SHAPthresh = 100
        Boolean converProId = true
        String CorMethod = "spearman"
        Float CorThreshold = 0.8
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
        if ~{converProId}; then
            converProId="TRUE"
        else
            converProId="FALSE"
        fi
        Rscript /scripts/protein_network.R \
            --score_thresholdHere ~{score_thresholdHere} \
            --combined_score_thresholdHere ~{combined_score_thresholdHere} \
            --SHAPthresh ~{SHAPthresh} \
            --patternChosen ${patternChosen} \
            --converProId ${converProId} \
            --proteinExpFile ~{proteinExpFile} \
            --CorMethod ~{CorMethod} \
            --CorThreshold ~{CorThreshold}
        
        tar -czvf ~{output_prefix}_pro_net_results.tar.gz --ignore-failed-read *.{png,*.xlsx}
    >>>
    output {
        File results = output_prefix + "_pro_net_results.tar.gz"
    }
    runtime {
        docker: "~{docker}"
        cpu: "~{cpu}"
        memory: "~{memory_gb}GB"
        disks: "local-disk 20 HDD"
    }
}