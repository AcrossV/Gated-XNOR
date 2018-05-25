# GXNOR-Net
GXNOR-Net: Training deep neural networks with ternary weights and activations without full-precision memory under a unified discretization framework
(https://www.sciencedirect.com/science/article/pii/S0893608018300108)

By Lei Deng, Peng Jiao, Jing Pei, Zhenzhi Wu, Guoqi Li

Abstrct: Although deep neural networks (DNNs) are being a revolutionary power to open up the AI era, the notoriously huge hardware overhead has challenged their applications. Recently, several binary and ternary networks, in which the costly multiply-accumulate operations can be replaced by accumulations or even binary logic operations, make the on-chip training of DNNs quite promising. Therefore there is a pressing need to build an architecture that could subsume these networks under a unified framework that achieves both higher performance and less overhead. To this end, two fundamental issues are yet to be addressed. The first one is how to implement the back propagation when neuronal activations are discrete. The second one is how to remove the full-precision hidden weights in the training phase to break the bottlenecks of memory/computation consumption. To address the first issue, we present a multi-step neuronal activation discretization method and a derivative approximation technique that enable the implementing the back propagation algorithm on discrete DNNs. While for the second issue, we propose a discrete state transition (DST) methodology to constrain the weights in a discrete space without saving the hidden weights. Through this way, we build a unified framework that subsumes the binary or ternary networks as its special cases, and under which a heuristic algorithm is provided at the website https://github.com/AcrossV/Gated-XNOR. More particularly, we find that when both the weights and activations become ternary values, the DNNs can be reduced to sparse binary networks, termed as gated XNOR networks (GXNOR-Nets) since only the event of non-zero weight and non-zero activation enables the control gate to start the XNOR logic operations in the original binary networks. This promises the event-driven hardware design for efficient mobile intelligence. We achieve advanced performance compared with state-of-the-art algorithms. Furthermore, the computational sparsity and the number of states in the discrete space can be flexibly modified to make it suitable for various hardware platforms.

Recommendation：  
Python 2.7  
Numpy 1.11.1  
Scipy 0.18.1    
cuda7.5  
cuDNN v4  
Theano 0.8 (Setting your Theano flags to use the GPU)   
（Warnings occur if using Theano 0.9）   
Pylearn2  
Lasagne 0.2.dev1  
The datasets you need  
A fast Nvidia GPU

Others : We focus on best test error rate not test error rate on datasets.    
Besides, the lr configuration on CIFAR10 adopts the widely used lr decay at specific steps.    
If you want to disable GXNOR-Net, please set discrete=False and use a smaller batch_size as that in usual full-precision networks.

Please cite our paper if you use this code: https://www.sciencedirect.com/science/article/pii/S0893608018300108

