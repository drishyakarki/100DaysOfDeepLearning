# 100 Days of Deep Learning

![Image](images/deep-neural-networks.webp)

🚀 Hello everyone, I'm excited to share that I'm diving into #100DaysOfDeepLearning to strengthen my foundations as well as help others who are wondering where to, how to start their Deep Learning journey.

💬 I will be starting this journey from the basics of DL so that any one with minimum knowledge regarding machine learning can follow. So, I will start with the famous Andrew NG's Deep Learning Specialization Course along with the book Deep Learning for Coders with Fastai and PyTorch.


*Note: I have studied about these topics in detail but obviously, it is not feasible to include all the things I learnt here. So, I hope that you will also learn these things from the course and the book yourself thoroughly.*

💡 If you want to join this amazing world of machine learning and deep learning, but don't know how, you can check out this amazing medium blog by [**Aleksa Gordić**](https://gordicaleksa.medium.com/get-started-with-ai-and-machine-learning-in-3-months-5236d5e0f230)

## Resources
- [**Deep Learning Specialization**](https://www.coursera.org/specializations/deep-learning)

## Books
- [**Deep Learning for Coders with Fastai and PyTorch**](https://www.amazon.com/_/dp/1492045527?smid=ATVPDKIKX0DER&_encoding=UTF8&tag=oreilly20-20)
- [**Dive into Deep Learning**](https://d2l.ai/)

## Projects and Notebooks
1. [**Dive Into Deep Learning**](https://github.com/drishyakarki/Dive-into-Deep-Learning)
-------

## Day 1

Starting our #Day1 of #100DaysOfDeepLearning, here are the few things I learned today from the course:

- Neural Network: Basically an architecture which consists of nodes(or neurons), layers and weights. Basically, input nodes receive information, hidden layers process it, and output nodes produce the final result.

- Then I learned about supervised learning with neural network, their applications. There are: (a)Structured Data - eg: Housing Data (b)Unstructured data - eg: text, audio, image.

- Later in the course, Andrew explains about why deep learning is being popular more and more. He uses the following graph to illustrate it.

![](images/plateau-data-models.png)

- He talks about how there are more data, more better computational hardware and  better research algorithm which will all contribute in significant amount of advancements in Deep Learning field for a long time in future too.

**Resources**
- [**Deep Learning Specialization**](https://www.coursera.org/specializations/deep-learning)

## Day 2

Continuing the day 2 of my 100DaysOfDeepLearning, today I learned about binary classification and how to apply logistic regression for the binary classification. I learned about the need of sigmoid function, how to learn weight and biases for accurate predictions. Furthermore, I learned about loss function and cost function of logistic regression - the main difference between them. I have posted a short summary of the things I learned in the linkedin and also the python implementation. You can check it out [**here**](https://www.linkedin.com/posts/drishya-karki_day2-activity-7130925730271547392-KOjz?utm_source=share&utm_medium=member_android) I hope you will also give your time to study about these topics following the Deep Learning Specialization course. Looking forward to the days ahead! [**Deep Learning Specialization**](https://www.coursera.org/specializations/deep-learning).

![day2](images/day2.png)

**Resources**
- [**Deep Learning Specialization**](https://www.coursera.org/specializations/deep-learning)

## Day 3

On the third day of #100daysofdeeplearning, today I learned about gradient descent, how it is derived and also implemented logistic regression gradient descent in python. I also learned about computation graph and derivatives with computation graph.I have posted a short summary of the things I learned in the linkedin. You can check it out [**here**](https://www.linkedin.com/posts/drishya-karki_day3-activity-7131293446408966144-9P_I?utm_source=share&utm_medium=member_desktop). I hope you will also spend some time to study about these topics following the [**Deep Learning Specialization**](https://www.coursera.org/specializations/deep-learning) course. Looking forward to the days ahead! 

![day3](images/day3.png)

**Resources**
- [**Deep Learning Specialization**](https://www.coursera.org/specializations/deep-learning)

## Day 4

Today I learned about vectorization - an important technique which helps in performing operations quite faster rather than the typical **for loops**. It enables parallelization on both CPU and GPU. I learned about its importance, how it can help in reducing computational cost and make it more efficient, implemented it for forward propagation as well as backward propagation of logistic regression. It is recommended to implement vectorization rather than explicit for loops whenever possible. It introduces SIMD(Single Instruction Multiple Data). Below are the python implementation of the things I learned. Excited for the coming days! Happy Learning :)

**Basic Implementation showing the difference between the computation of vectorized and non-vectorized approach**
![day4](images/day4.png)

**In Logistic Regression**
![day4](images/day4LR.png)

**Resources**
- [**Deep Learning Specialization**](https://www.coursera.org/specializations/deep-learning)

## Day 5

**Broadcasting** is a very important concept to understand in numpy which is very useful for performing mathematical operations between arrays of different shapes. Continuing my journey on the 5th day, I learned about broadcasting and its role, common steps for preprocessing a new dataset, how to build parts of your algorithms, helper functions, forward and backward propagation along with optimization. Below is the python implementation of the things I learned. I have also started learning from the book [**Dive into Deep Learning**](https://d2l.ai/). Excited for the coming days! Happy Learning :)

![day5](images/day5.png)

**Resources**
- [**Deep Learning Specialization**](https://www.coursera.org/specializations/deep-learning)
- [**Broadcasting in Numpy**](https://numpy.org/doc/stable/user/basics.broadcasting.html)

## Day 6

🚀 Activation functions are mathematical operations applied to the output of the neuron in neural network which introduces non-linearities to the network. It enables a network to learn complex patterns and relationships in the data. 
On the Day-6 of #100DaysOfDeepLearning, today I learned about various activation functions - sigmoid, tanh, ReLU, leaky ReLU and the intuition behind them.I also studied about the gradient descent for neural network, cost function and random initilaization of the neural network. From the book D2L, I learned about object oriented design for implementation. At a high level, we wish to have three classes: 
(i) Module 
(ii) DataModule
(iii) Trainer 
Hope you will also spend some time to study about these topics following the Deep Learning Specialization course and Dive into Deep Learning book. Looking forward to the days ahead! Happy Learning :)

**Object Oriented Design Implementation**
![day6](images/day6.png)

**Sin-Cos plot with different smoothness**
![day6](images/day6-sinecos.png)

**Resources**
- [**Deep Learning Specialization**](https://www.coursera.org/specializations/deep-learning)
- [**Dive into Deep Learning**](https://d2l.ai/)

## Day 7

Today I learned about deep l-layer neural network, forward and backward propagation on deep network, how to get your dimensions right, building blocks of DNN and about the parameters and hyperparameters from **Deep Learning Specialization**. From the book **Dive into Deep Learning** I read about different concepts such as model complexity, underfitting or overfitting, polynomial curve fitting, cross-validation, loading the dataset, reading the minibatch, softmax, cross-entropy loss. Below is just the snapshot of the implementation of softmax regression from scratch and also the concise implentation of softmax regression along with the visualization of fashionMNIST. You can checkout the notebooks for the full implementation.

![day7](images/day7.png)

**Resources**
- [**Deep Learning Specialization**](https://www.coursera.org/specializations/deep-learning)
- [**Dive into Deep Learning**](https://d2l.ai/)
- [**ImageClassificationDataset**](https://github.com/drishyakarki/Dive-into-Deep-Learning/blob/main/imageClassificationDataset.ipynb)
- [**Softmax Regression**](https://github.com/drishyakarki/Dive-into-Deep-Learning/blob/main/softmaxRegression.ipynb)

## Day 8

A Multi-Layer Perceptron is a type of artificial neural network that consists of multiple layers of nodes, or neurons, organized in a series of interconnected layers. It is a feedforward neural network.On the Day-8 of #100daysofdeeplearning, today I learned about setting up ML Application, train/dev/test sets, bias and variance, the basic "recipe" for machine learning. I also studied about the bias-variance tradeoff, regularization. Also, from the book D2L, I studied about MLPs, universal approximators.
Below is the snapshot of the python implementation of the MLPs- you can checkout the full notebooks in the github repo below. I hope you will also spend some time dwelling on these topics from the book and the specialization course itself. Happy Learning :)

![day8](images/day8.png)

**Resources**
- [**Deep Learning Specialization**](https://www.coursera.org/specializations/deep-learning)
- [**Dive into Deep Learning**](https://d2l.ai/)
- [**Implementation of MultiLayer-Perceptron**](https://github.com/drishyakarki/Dive-into-Deep-Learning/blob/main/mlpImplementation.ipynb)

## Day 9

On the ninth day of #100daysofdeeplearning, today I learned about various topics such as regualarization, how it helps in reducing variance and preventing overfitting, different regularization techniques - L1 regularization, L2 regularization, dropout, weight decay(sometimes L2-Reg is also called weight decay) and data augmentation from the [**Deep Learning Specialization**](https://www.coursera.org/specializations/deep-learning). Similarly, I learned about vanishing gradients, exploding gradients, visualized them; numerical stability and the need for correct parameter initialization, early stopping, implemented dropout and also predicted housing prices on kaggle: data preprocessing, error measure, k-fold cross validation, weight decay. 
Below is the snapshot of the code- you can checkout the full notebooks by visiting the links below. I hope you will also spend some time dwelling on these topics from the book and the specialization course itself. Happy Learning :)

![day9](images/day9.png)
![day9](images/day9-2.png)

**Resources**
- [**Deep Learning Specialization**](https://www.coursera.org/specializations/deep-learning)
- [**Dive into Deep Learning**](https://d2l.ai/)
- [**Predicting House Prices**](https://github.com/drishyakarki/Dive-into-Deep-Learning/blob/main/kaggleHousePrediction.ipynb)

## Day 10

On the Day-10 of #100daysofdeeplearning, I learned about numerical approximations of gradients, gradient checking, understanding mini-batch gradient descent, various optimization algorithms such as exponentially weighted average,bias correction, gradient descent with momentum, RMSProp and the combination of these two Adam Optimization algorithm from the [**Deep Learning Specialization**](https://www.coursera.org/specializations/deep-learning). Similarly, I read and implemented different topics from the book [**Dive into Deep Learning**](https://d2l.ai/)(chapter: **Builder's Guide**) relating to construction of custom models, creating custom layers and modules, parameter management, accessing the targeted parameters, tied parameters, layers with and without parameters, reading and writing tensors to file and gpu usage.
Below is just the some part of the code from all the notebooks- you can checkout the full notebook implementation in the github repo. I hope you will also spend some time dwelling on these topics from the book and the specialization course itself. Happy Learning :)

![day10](images/day10.png)

**Resources**
- [**Deep Learning Specialization**](https://www.coursera.org/specializations/deep-learning)
- [**Dive into Deep Learning**](https://d2l.ai/)
- [**Builder's Guide**](https://github.com/drishyakarki/Dive-into-Deep-Learning/tree/main/buildersGuide)

## Day 11

On the Day-11 of #100daysofdeeplearning, I learned about various topics relating to Convolutional Neural Networks such as translation invariance, locality, convolution, feature map, receptive layers, padding and strides, batch normalization, LeNet architecture. I also studied and implemented different modern CNN architectures: AlexNet, VGG, Network in Network, googleNet, ResNet, DenseNet, RegNet.
Below is the architecture of LeNet and ResNet from all the notebooks- you can checkout the full notebook for the implementation of other networks in the github repo. I have two seperate folders, one for CNN in low level from scratch, and other, with the modern CNNs architecture I hope you will also spend some time dwelling on these topics from the book Dive into Deep Learning.Happy Learning :)

![day11](images/day11.png)

**Resources**
- [**Dive into Deep Learning**](https://d2l.ai/)

**Notebooks**
- [**CNN**](https://github.com/drishyakarki/Dive-into-Deep-Learning/tree/main/cnn)
- [**Modern CNN**](https://github.com/drishyakarki/Dive-into-Deep-Learning/tree/main/modernCNN)

## Day 12

On the Day-12 of #100daysofdeeplearning, I learned about sequence models, vocabulary, recurrent neural network: forward propagation and backward propagation through time, different types of RNN, language model and sequence generation, autoregressive models, markov models, the order of decoding, converting raw text into sequence data, tokenization, exploratory language statistics, zipf's law, unigram, bigram and trigram.
Below is the snapshot of the code- you can checkout the full notebook for the implementation of other networks in the github repo.I hope you will also spend some time dwelling on these topics from the book Dive into Deep Learning.Happy Learning :)

![day12](images/day12.png)

**Resources**
- [**Dive into Deep Learning**](https://d2l.ai/)
- [**Deep Learning Specialization**](https://www.coursera.org/specializations/deep-learning)

**Notebooks**
- [**Sequence models**](https://github.com/drishyakarki/Dive-into-Deep-Learning/tree/main/RNN)

## Day 13

Today I learned about language models, sequence generation, vanishing gradients with RNN, exploding gradients with RNN, word embeddings, laplace smoothing, perpelxity, RNN. I also learned and implemented RNN from scratch along with gradient clipping, one hot encoding and decoding. 
Here is the snapshot of the code. I have uploaded the full implementation notebook in the github, you can check it out if you like. Hope you will also spend some time reading about these topics. Keep Learning :)

![day13](images/day13.png)

**Resources**
- [**Dive into Deep Learning**](https://d2l.ai/)
- [**Deep Learning Specialization**](https://www.coursera.org/specializations/deep-learning)

**Notebooks**
- [**RNN from scratch**](https://github.com/drishyakarki/Dive-into-Deep-Learning/tree/main/RNN)

## Day 14

Our from-scratch implementations was designed to provide insight into how each components work. But for writing production code and using RNNs every day, we will want to rely more on libraries that provide us efficiency in both implentation time as well as computation time. Today, on the day 14th of #100daysofdeeplearning, I learned and implemented RNNs concisely by high-level APIs on the Time Machine Dataset.
Here is the snapshot of the code. You can check out the results and predictions in the github repo itself. Hope you will also spend some time reading about these topics. Keep Learning :)

![day14](images/day14.png)

**Resources**
- [**Dive into Deep Learning**](https://d2l.ai/)

**Notebooks**
- [**Concise Implementation of RNN**](https://github.com/drishyakarki/Dive-into-Deep-Learning/tree/main/RNN)

