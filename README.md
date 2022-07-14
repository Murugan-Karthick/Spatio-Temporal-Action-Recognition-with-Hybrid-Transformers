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

### Build Transformer Encoder
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
