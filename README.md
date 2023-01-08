# Convolutional Neural Network Using PyTorch   

implementing a convolutional neural network (CNN) and training it on the MNIST dataset, to recognize handwriting digits.  

The CNN architecture I used:  
Conv layer (10 5x5 Kernels) -> Max Pooling (2x2 kernel) -> Relu -> Conv layer (20 5x5 Kernels) -> Max Pooling (2x2 kernel) -> Relu -> Hidden layer (320 units) -> Relu -> Hidden layer (50 units) -> Output layer (10 outputs).

I used the MNIST dataset which consists of greyscale handwritten digits images.
Each image is 28x28 pixels and there are 10 different digits.  
The network will take these images and predict the digit in them.

<p float="left">
<img width="500" alt="image" src="https://user-images.githubusercontent.com/112930532/211188122-9b5d41ff-eb3f-44dd-b94b-b2aee58412d7.png">
<img width="350" alt="image" src="https://user-images.githubusercontent.com/112930532/211188147-cbb7cbad-5c2b-4007-830e-70e147470f79.png"> 

The Training Process:
<img width="300" alt="image" src="https://user-images.githubusercontent.com/112930532/211188061-d484ee21-e569-4422-b1f6-062a285e089b.png">  

The best Hyper parameters that achieve the best accuracy (98% !):
<img width="300" alt="image" src="https://user-images.githubusercontent.com/112930532/211188068-6b1b3028-120a-483c-8f63-79f813164439.png">
