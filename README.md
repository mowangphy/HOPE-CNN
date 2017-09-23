# HOPE-CNN
Using HOPE model to train CNNs
Without data augmentation: 7.06% error rate on the cifar-10 validation set and 29.47% error rate on the cifar-100 validation set (Single-HOPE-Block)

With data augmentation (rotation+translation+scale+cololr casting): 5.89% error rate on the cifar-10 validation set and 26.99% error rate on the cifar-100 validation set (Single-HOPE-Block).

The paper has been accepted by ACML 2017.

If you hope to use this code, please cite:

@article{pan2016learning,

  title={Learning Convolutional Neural Networks using Hybrid Orthogonal Projection and Estimation},
  
  author={Pan, Hengyue and Jiang, Hui},
  
  journal={arXiv preprint arXiv:1606.05929},
  
  year={2016}
  
  }

Based on MatConvNet

**MatConvNet** is a MATLAB toolbox implementing *Convolutional Neural
Networks* (CNNs) for computer vision applications. It is simple,
efficient, and can run and learn state-of-the-art CNNs. Several
example CNNs are included to classify and encode images. Please visit
the [homepage](http://www.vlfeat.org/matconvnet) to know more.
