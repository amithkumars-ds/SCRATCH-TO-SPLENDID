# BUILDING FROM SCRATCH

A from scratch implementation of machine learning, deep learning, and a few detours into the math and physics that make it all work. Everything here uses bare-minimum libraries (mostly `numpy`) — no `sklearn.fit()`, no `torch.nn.Module` shortcuts. The goal is to understand every algorithm well enough to write it from first principles, in the spirit of building in public.


## ML from Scratch

### Supervised Learning

**Regression**
- [x] [Linear Regression](./ML%20from%20Scratch/Supervised%20Learning/1_linear_regression.py)
- [ ] Ridge Regression
- [ ] Lasso Regression
- [ ] Elastic Net
- [ ] Polynomial Regression
- [x] [k-Nearest Neighbors Regressor (kNN)](./ML%20from%20Scratch/Supervised%20Learning/4_knn_regression.py)
- [ ] Support Vector Machine Regressor (SVM)
- [ ] Decision Tree Regressor
- [ ] Random Forest Regressor
- [ ] XGBoost Regressor

**Classification**
- [x] [Logistic Regression](./ML%20from%20Scratch/Supervised%20Learning/2_logistic_regression.py)
- [x] [k-Nearest Neighbors Classifier (kNN)](./ML%20from%20Scratch/Supervised%20Learning/3_knn_classifier.py)
- [ ] Naive Bayes
- [ ] Support Vector Machine Classifier (SVM)
- [ ] Decision Tree
- [ ] Random Forest
- [ ] Gradient Boosting (from scratch)
- [ ] AdaBoost
- [ ] XGBoost-style boosted trees

### Unsupervised Learning

**Clustering**
- [ ] K-Means
- [ ] Hierarchical Clustering
- [ ] DBSCAN
- [ ] Gaussian Mixture Models (GMM)

**Dimensionality Reduction**
- [ ] PCA
- [ ] t-SNE
- [ ] Linear Discriminant Analysis (LDA)
- [ ] Autoencoders 

**Association Rule Learning**
- [ ] Apriori
- [ ] FP-Growth
- [ ] Eclat

### Reinforcement Learning
- [ ] Multi-Armed Bandits
- [ ] Markov Decision Processes (value/policy iteration)
- [ ] Q-Learning
- [ ] SARSA
- [ ] Deep Q-Network (DQN)
- [ ] Policy Gradient (REINFORCE)
- [ ] Actor-Critic

---

## Deep Learning from Scratch

- [ ] Perceptron
- [ ] ANN (Multi-Layer Perceptron + Backpropagation)
- [ ] CNN (Convolutional Neural Network)
- [ ] RNN (Recurrent Neural Network)
- [ ] LSTM (Long Short-Term Memory)
- [ ] Bidirectional LSTM
- [ ] GRU (Gated Recurrent Unit)
- [ ] Attention Mechanism
- [ ] Transformer (Encoder-Decoder from scratch)
- [ ] GPT-like Model (Karpathy-style build-up: bigram → nanoGPT)

## Math & Misc

- [ ] Linear Algebra Essentials (vectors, matrices, eigenvalues/eigenvectors)
- [ ] Probability & Statistics Primers
- [ ] Calculus for Backpropagation (gradients, chain rule)
- [ ] Optimization Algorithms (SGD, Momentum, RMSProp, Adam)
- [ ] Information Theory (Entropy, KL Divergence)
- [ ] Quantum Mechanics Basics (exploratory notes)


## Philosophy

> No black boxes. Everything is built from scratch — until it's splendid.

