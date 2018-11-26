# Blind Video Quality Assessment with Weakly Supervised Learning and Resampling Strategy

This is an implementation of "Blind Video Quality Assessment with Weakly Supervised Learning and Resampling Strategy." 

 And this document will walk you through all the files contained in the package.

Note that this implement doesn't include the resampling strategy because I didn't fully understand that part.

## FILES

### ms_ssim

This folder contains the implementation of "Multi-scale Structural Similarity Index (MS-SSIM)". We use it to evaluate video block's reference-distorted pair difference and generate the soft label.

### traverse_database.m

MATLAB script, iterate through a given database, call `generate_block.m` to generate video block representation and their soft label.

### generate_block.m

MATLAB function. Generate video block and soft label for a given video series.

Function form:

```matlab
function generate_block(ref_name, dis_name, dims, block_size, stride, output_pattern)

% ref_name: reference video name
% dis_name: distorted video name
% dims: video size info, [height, width]
% block_size: block size info, [n1, n2, n3]
% stride: step size in each video dimension when splitting video blocks
% output_pattern: video block storage name format
```

### data_processor.py

Python script. Create **PyTorch dataset** for fast training. Including two classes:

**BlockSet**: Create video block dataset.

**FHSet**: Create frequency histogram dataset. This dataset is used to provide data for linear regression.

### model.py

Python script. Implemented the CNN architecture proposed in the original paper and a linear regression model.

It contains three classes:

```python
class ArcNet(nn.Module)
"""
This class is the main architecture of the convolution network
:param is_training: define the phase of the model
"""

class CNNBlock(nn.Module)
"""
This class implement a cnn block
:param in_channels: input dimensions
:param out_channels: output dimensions
:param kernel_size: conv kernel size
:param stride: conv stride
:param padding: padding size
:param batchnorm: bool, use batchnorm?
:param relu: bool, use relu?
:param max_pooling: bool, using max_pooling?
:param pooling_kernel: max_pooling kernel size
:param is_training: model phase
"""

class LinearRegression(nn.Module)
"""
Linear regression
:param in_channels: input dimensions
"""
```



### trainer.py

Train the model

> batch_size: 512
>
> max epoch: 100
>
> learning rate: 0.0005
>
> optimizer: Adam

### utils.py

Utility class. 



