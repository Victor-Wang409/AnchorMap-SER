## Example Usage

```bash
# Stage I: Finetuning WavLM on Emotion Classification
python wavLM.py --data ./data/audio_partial5_train_dataset.pickle \
                --save-path ./dump \
                --mode train \
                --name train \
                --batch-size 16 \
                --num-labels 5 \
                --num-epochs 21 \
                --write

# Stage I: Performing Inference on Finetuned WavLM to obtain emotion features
python wavLM.py --data ./data/audio_partial5_train_dataset.pickle \
                --mode inference \
                --num-labels 5 \
                --load-path ./dump/train/model_best.pth \
                --save-path ./dump \
                --batch-size 16 


# Stage II: Fitting the Anchored Dimensionality Reduction and Transforming New Data
reducer = AVLearner()
train_y = reducer.fit_transform(embedding, label, init_global)
test_y = reducer.transform(test_embedding)
```

## Citation