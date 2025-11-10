#!/bin/bash
# Nsight Systems profiling script for CUDA Inference Engine
# Usage: ./profile.sh [executable] [output_name]

EXECUTABLE=${1:-./build/cuda_inference_engine}
OUT_NAME=${2:-cuda_profile}

echo "Running Nsight Systems profiling..."
nsys profile \
    --stats=true \
    --force-overwrite=true \
    --trace=cuda,nvtx,osrt,driver \
    --sample=cpu \
    -o ${OUT_NAME} \
    ${EXECUTABLE}

echo "Profile complete!"
echo "Output file: ${OUT_NAME}.qdrep"
echo "Open in Nsight Systems GUI or convert to report:"
echo "nsys stats ${OUT_NAME}.qdrep"


#Make it executable:

#chmod +x profile.sh


#Run it:

#./profile.sh ./build/cuda_inference_engine orin_profile
