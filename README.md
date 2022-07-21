# Action Recognition with Transformers

**Training a action recognizer with hybrid transformers.**

We will build a Transformer-based model to recognize the actions from videos.we are going to develop hybrid Transformer-based models for action recognition that operate on CNN feature maps.

## Download the dataset
In order to make training time to low, we will be using a subsampled version of the original [UCF101 dataset](https://www.crcv.ucf.edu/data/UCF101.php) dataset. download the dataset from [UCF101](https://git.io/JGc31) dataset link.

## Requirements
Before run the code you should run below the lines for installing dependencies
```bash
  pip install tensorflow
  pip install -q git+https://github.com/tensorflow/docs
  pip install imutils
  pip install opencv-python
  pip install matplotlib
  pip install seaborn
```

## Data preparation

We will mostly be following the same data preparation steps in this example, except for
the following changes:

* We took the image size to 128x128 to speed up computation ``IMG_SIZE = 128``.
* We use [DenseNet121](http://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf)
for feature extraction.
* We directly pad shorter videos by zero to length `MAX_SEQ_LENGTH`.

## Feature extractor

we use densenet to extract the meaningful features from the frames
```
def build_feature_extractor():
    feature_extractor = keras.applications.DenseNet121(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.densenet.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")


feature_extractor = build_feature_extractor()
```

## Dependant variable preprocessing

we used StringLookup to encode dependant (or) class variables
```
label_processor = keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(train_df["tag"]), mask_token=None
)
print(label_processor.get_vocabulary())
```

## Building the Transformer-based model

First, self-attention layers that form the basic blocks of a Transformer are
order-agnostic. Instead of focusing all the frames the attention mechanism focus important portion.

<img src="https://miro.medium.com/max/1400/1*iy12bH-FiUNOy9-0bULgSg.png"/>
<p align="center">
    <b>Transformers Encoder and Decoder Architecture</b>
</p>

### Positional embedding
Since videos are ordered sequences of frames, we need our
Transformer model to take into account order information.
We do this via **positional encoding**.
We simply embed the positions of the frames present inside videos with an
[`Embedding` layer] and it will results `postional embeddings`. We then
add these positional embeddings to the precomputed CNN feature maps.
```
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        # embedding layer to generate positional embeddings
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )
```

### Building Transformer Encoder
We used only the encoder part of the transformer to perform action recognition

```
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        ....
        .....
        .......

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        
        # step1: positional embedding is input to self attention layer
        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        # step2: the output of attention and positional embedding is input to layer normalization1 layer
        proj_input = self.layernorm_1(inputs + attention_output)
        # step3: the output of layer normalization layer1 is input to dense layer
        proj_output = self.dense_proj(proj_input)
        # step4: the output of dense layer and the output of layer normalization layer1 input to layer mormalization2 layer
        # the output of layer normalization2 is passed to decoder
        return self.layernorm_2(proj_input + proj_output)
```

## Model training
```
def run_experiment():
    filepath = "./tmp/action_recognizer"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=1
    )

    model = get_compiled_model()
    history = model.fit(
        train_data,
        train_labels,
        validation_split=0.15,
        epochs=EPOCHS,
        callbacks=[checkpoint],
    )

    model.load_weights(filepath)
    _, accuracy = model.evaluate(test_data, test_labels)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return history, model
```

## Model performance
<img src="https://github.com/Murugan-Karthick/Spatio-Temporal-Action-Recognition-with-Hybrid-Transformers/blob/main/results.png" width="500" height="500"/>

## Inference
For inference we need to do video preprocessing before input to the model
```
def prepare_single_video(frames):
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    # Pad shorter videos.
    if len(frames) < MAX_SEQ_LENGTH:
        diff = MAX_SEQ_LENGTH - len(frames)
        padding = np.zeros((diff, IMG_SIZE, IMG_SIZE, 3))
        frames = np.concatenate(frames, padding)

    frames = frames[None, ...]

    # Extract features from the frames of the current video.
    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            if np.mean(batch[j, :]) > 0.0:
                frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
            else:
                frame_features[i, j, :] = 0.0

    return frame_features
```

## Check sample predictions
```
def predict_action(path):
    class_vocab = label_processor.get_vocabulary()

    frames = load_video(os.path.join("test", path))
    frame_features = prepare_single_video(frames)
    probabilities = trained_model.predict(frame_features)[0]

    for i in np.argsort(probabilities)[::-1]:
        print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
    return frames
```

## Output
<img src="https://github.com/Murugan-Karthick/Spatio-Temporal-Action-Recognition-with-Hybrid-Transformers/blob/main/animation.gif" width="300" height="300"/>
PlayingCello: 100.00% <br>
ShavingBeard:  0.00% <br>
Punch:  0.00% <br>
TennisSwing:  0.00% <br>
CricketShot:  0.00%
