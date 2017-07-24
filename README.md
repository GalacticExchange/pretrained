# Pretrained

The most complete and frequently updated list of pretrained top-performing models. Tensorflow, Theano and others. 

Want to add your model? File an issue, and we will add it.

## Image recognition

TensorFlow model file, the link to the model checkpoint, and the top 1 and top 5
accuracy (on the imagenet test set).
Note that the VGG and ResNet V1 parameters have been converted from their original
caffe formats
([here](https://github.com/BVLC/caffe/wiki/Model-Zoo#models-used-by-the-vgg-team-in-ilsvrc-2014)
and
[here](https://github.com/KaimingHe/deep-residual-networks)),
whereas the Inception and ResNet V2 parameters have been trained internally at
Google. Also be aware that these accuracies were computed by evaluating using a
single image crop. Some academic papers report higher accuracy by using multiple
crops at multiple scales.

Model | Url  | 
:----:|:------------:|
[Inception V1](http://arxiv.org/abs/1409.4842v1)| |https://github.com/tensorflow/models/tree/master/slim#Pretrained
[Inception V2](http://arxiv.org/abs/1502.03167)|https://github.com/tensorflow/models/tree/master/slim#Pretrained
[Inception V3](http://arxiv.org/abs/1512.00567)|https://github.com/tensorflow/models/tree/master/slim#Pretrained
[Inception-ResNet-v2](http://arxiv.org/abs/1602.07261)| https://github.com/tensorflow/models/tree/master/slim#Pretrained
[ResNet V1 50](https://arxiv.org/abs/1512.03385)|https://github.com/tensorflow/models/tree/master/slim#Pretrained
[ResNet V1 101](https://arxiv.org/abs/1512.03385)|https://github.com/tensorflow/models/tree/master/slim#Pretrained
[ResNet V1 152](https://arxiv.org/abs/1512.03385)| https://github.com/tensorflow/models/tree/master/slim#Pretrained
[ResNet V2 50](https://arxiv.org/abs/1603.05027) | https://github.com/tensorflow/models/tree/master/slim#Pretrained
[ResNet V2 101](https://arxiv.org/abs/1603.05027)|https://github.com/tensorflow/models/tree/master/slim#Pretrained
[ResNet V2 152](https://arxiv.org/abs/1603.05027) | https://github.com/tensorflow/models/tree/master/slim#Pretrained 
[ResNet V2 200](https://arxiv.org/abs/1603.05027)|https://github.com/tensorflow/models/tree/master/slim#Pretrained
[VGG 16](http://arxiv.org/abs/1409.1556.pdf)|https://github.com/tensorflow/models/tree/master/slim#Pretrained

[vgg_16_2016_08_28.tar.gz](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz)|71.5|89.8| same |

[VGG 19](http://arxiv.org/abs/1409.1556.pdf)|[Code](https://github.com/tensorflow/models/blob/master/slim/nets/vgg.py)|
[vgg_19_2016_08_28.tar.gz](http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz)|71.1|89.8| same |

[MobileNet_v1_1.0_224](https://arxiv.org/pdf/1704.04861.pdf)|https://github.com/tensorflow/models/tree/master/slim#Pretrained
[MobileNet_v1_0.50_160](https://arxiv.org/pdf/1704.04861.pdf)|https://github.com/tensorflow/models/tree/master/slim#Pretrained
[MobileNet_v1_0.25_128](https://arxiv.org/pdf/1704.04861.pdf)|https://github.com/tensorflow/models/tree/master/slim#Pretrained
Pretrained ConvNets for pytorch: ResNeXt101, ResNet152, InceptionV4, InceptionResnetV2, etc. | https://github.com/Cadene/pretrained-models.pytorch

DenseNet | https://github.com/flyyufelix/DenseNet-Keras 


## Object detection

http://www.vlfeat.org/matconvnet/pretrained/#object-detection

## Face recognition

VGG-Face.  http://www.vlfeat.org/matconvnet/pretrained/#face-recognition

## Pix2Pix style transfer

Webcam Pix2Pix https://github.com/memo/webcam-pix2pix-tensorflow

## Image caption generation

Show and Tell : A Neural Image Caption Generator https://github.com/KranthiGV/Pretrained-Show-and-Tell-model


## Text translation, summarization etc

OpenNMT (http://opennmt.net/Models/)

Includes: English -> German, German -> English, English Summarization, Multi-way - FR,ES,PT,IT,RO<>FR,ES,PT,IT,RO, Image-to-text generation.

## Semantic segmentation

Semantic segmentation https://github.com/ndrplz/dilation-tensorflow

Semantic segmentation 2 http://www.vlfeat.org/matconvnet/pretrained/#semantic-segmentation


## Speech recognition

Baidu Warp-CTC  (https://github.com/SeanNaren/deepspeech.torch)




