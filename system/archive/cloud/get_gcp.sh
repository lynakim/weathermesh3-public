#!/bin/bash

create_instance() {
    ZONE=$1

    CMD="gcloud compute instances create gputest3 \
        --project=wbnwp-1 \
        --zone=$ZONE\
        --machine-type=a2-highgpu-8g \
        --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default \
        --no-restart-on-failure \
        --maintenance-policy=TERMINATE \
        --provisioning-model=STANDARD \
        --service-account=444839991314-compute@developer.gserviceaccount.com \
        --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append \
        --accelerator=count=8,type=nvidia-tesla-a100 \
        --tags=http-server,https-server \
        --create-disk=auto-delete=yes,boot=yes,device-name=gputest3,image=projects/ml-images/global/images/c0-deeplearning-common-cu121-v20231105-debian-11,mode=rw,size=2000,type=projects/wbnwp-1/zones/$ZONE/diskTypes/pd-ssd \
        --no-shielded-secure-boot \
        --shielded-vtpm \
        --shielded-integrity-monitoring \
        --labels=goog-ec-src=vm_add-gcloud \
        --reservation-affinity=any"

    echo -e "\033[33m$CMD\033[0m"
    eval $CMD
}

gcloud compute accelerator-types list --project=wbnwp-1 | grep "a100  " | awk '{print $3}' | while read -r zone; do create_instance "$zone"; done
