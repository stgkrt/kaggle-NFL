# TAG="kaggle_env"
# ENV_NAME="NFL"
# PROJECT_DIR="$(cd "$(dirname "${0}")" || exit; pwd)"

# docker run  --gpus all \
#             -p 8888:8888 \
#             -p 10022:22 \
#             --shm-size=48gb \
#             -it \
#             -v "${PROJECT_DIR}:/workspace" \
#             -w "/workspace" \
#             --name ${ENV_NAME} \
#             ${TAG} \
#             /bin/bash

TAG="kaggle_env"
PROJECT_DIR="$(cd "$(dirname "${0}")" || exit; pwd)"

docker run  --gpus all \
            -p 8888:8888 \
            -p 10022:22 \
            --shm-size=48gb \
            --rm -it \
            -v "${PROJECT_DIR}:/workspace" \
            -w "/workspace" \
            ${TAG} \
            /bin/bash
