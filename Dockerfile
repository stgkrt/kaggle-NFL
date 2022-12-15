FROM gcr.io/kaggle-gpu-images/python:v121

ENV lang="ja_jp.utf-8" language="ja_jp:ja" lc_all="ja_jp.utf-8"

# ADD run.sh /opt/run.sh
# RUN chmod 700 /opt/run.sh

RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install pytorch-lightning==1.7.6 \
		    pytorch-lightning-spells==0.0.3 \
		    wandb==0.12.9 \
		    python-box==5.4.1 \
		    faiss-gpu==1.7.2 \
		    python-box==5.4.1 \
		    grad-cam==1.3.1 \
		    ttach==0.0.3 \ 
		    mlflow==1.13.0 

