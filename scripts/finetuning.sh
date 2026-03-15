python wavLM.py --data ./data/audio_train_dataset.pickle \
                --save-path ./dump \
                --mode train \
                --name train \
                --batch-size 8 \
                --num-labels 7 \
                --num-epochs 21 \
                --name emodb \
                --write