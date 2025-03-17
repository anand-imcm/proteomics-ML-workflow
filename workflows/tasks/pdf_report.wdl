version 1.0

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