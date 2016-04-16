#**Tensorflow Projects**
A repo of everything deep and neurally related. Implementations and ideas are largely based on papers from arxiv and implementations, tutorials from the internet. 

 - Face Detection - Face Detection as a regression problem from kaggle.
 - ImageArt - Everything artistic with deep nets - 
	 - DeepDream,  LayerVisualization, NeuralStyle(Gatys et al), ImageInversion(Mahendran et al) - all implementations are VGG model based.
	 -  Generative NeuralStyle(Johnson et al, work in progress)
	 - NeuralArtist(a mapping from location to rgb as an optimization problem - idea based on karpathy's convnet.js implementation)
 - MNIST - My first ever code in Tensorflow. Check this out if you are new to Deep learning and Tensorflow - based on tensorflow tutorial with additions here and there. 
 - notMNIST - Well you got to follow up MNIST with something :D
 - logs - Tensorflow Summary and Saver directory for all problems.
 - There are a couple of more implementations as attempts to solve a few other problems
	 - YelpRestaturantClassification - Here's a tip: don't even bother trying this without a GPU.
	 - Deblurring - Posing blurring in images as conv net problem - architecture is based on Image super-resolution paper by Dong et al.
	 - FindInceptionSimilarity - This implementation made me realize an important concept in machine learning in general - Symbolism vs Distributed representations. 
	 - TensorflowUtils -Since most of the time parameters are just given a BKM value.

Here are a few results,
 - Deep dreaming a cloud
    
<img src="https://github.com/shekkizh/TensorflowProjects/blob/master/logs/Deepdream_logs/checkpoints/run3/0_clouds_deepdream_conv5_1.jpg" width="600" height="450" />

 - Visualizing the first filter of VGG layer conv5_3
 
<img src="https://github.com/shekkizh/TensorflowProjects/blob/master/logs/Visualization_logs/visualization_conv5_3_0.jpg" />

- NeuralArtist - Not exactly the best the network could do - but since my laptop was crying so loud, had to stop halfway through the optimization :(

<img src="https://github.com/shekkizh/TensorflowProjects/blob/master/logs/NeuralArtist_logs/run6/neural_artist_check_final.jpg" />

 
