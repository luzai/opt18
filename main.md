#

###      Motivation

- Multilayer networks are more powerful than single layer nets
    - \eg, XOR problem

\centering 

![](2018-03-10-11-05-54.png){width=70%} \


### Power of nonlinearity

- For linear neurons, a multilayer net is equivalent to
    a single-layer net. This is not the case for nonlinear
    neurons
    -  Why?





### MLP architecture

\centering 

![](2018-03-10-11-08-20.png){width=80%} \


### Multi-layer perceptron

- Think of an MLP as a complicated, non-linear
    function of its input parametrized by $\vw$:
    $$\vy=F(\vx;\vw)$$
-   Note that "Multi-layer perceptron" is a bit of a
    misnomer because they use a continuous activation
    function

### MLP Training

- Given a set of training data $\{\vx_p,\vd_p\}$  can we adjust
    $\vw$ so that the network is optimal
-   Optimal \wrt what criterion
    - Must define error criterion between $\vy _\vp = F(\vx_\vp ; \vw)$ and $\vd_p$
    - We will use the mean square error for now, but others are
        possible (and often preferable)
- Goal find $\vw$ that minimizes
$$  \bar{E}(\vw) = \sum_p E_p(\vw) = \sum_p \frac{1}{2} \|\vd_p - F(\vx_p ;\vw)\|^2$$

<!-- ### Backpropagation

Backpropagation will be continued on Thursday, March 18.  -->


### Backpropagation

- Because $\bar{E} (\vw)$ is still a complication non-linear
    function, we will optimize it using gradient descent
-   Because of the structure of MLPs, we can compute
    the gradient of $E_p (\vw)$ very efficiently using the
    backpropagation algorithm
-   Backpropagation computes the gradient of each
    layer recursively based on subsequent layers
-   Because this is $E_p (\vw)$ and not $\bar{E} (\vw)$, we will be
    using stochastic gradient descent


### Notation

- Notation for one hidden layer (drop p for now)

\centering

![](2018-03-10-13-48-45.png){width=70%} \ 



### Notation

- Notation for one hidden layer (drop p for now)

\centering 

![](2018-03-10-13-49-42.png){width=70%} \

\begin{eqnarray*}
   y_k &=& \varphi_k \left(\sum_j w_{kj} \varphi_j \left(\sum_i w_{ji} x_i \right) \right) \\
   E(\vw) &=& 1/2 \sum_k (d_k - y_k)^2  
\end{eqnarray*}

- Keep in mind during the derivation:
    - How would changing $E_p (\vw)$ affect the derivation
    - How would changing  $\varphi (\vv)$ affect the derivation


### Backprop


\centering 

![](2018-03-10-13-49-42.png){width=70%} \


\begin{align} 
	y & = \frac{1}{2} \sum_k (d_k - y_k)^2 \\ %\nonumber
      & =  \frac{1}{2}\sum_k \left(
        d_k - \underbrace{\varphi_k 
        \left( \overbrace{\sum_j w_{kj} y_j}^{v_k} \right)
         }_{y_k}
      \right)^2 
\end{align}

- Then, to adjust the hidden-output weights

\begin{equation}
    \frac{\partial }{\partial w_{kj}} E(\vw) = 
    \textcolor{red}{
        \frac{\partial E}{\partial y_k}} 
    \textcolor{blue}{
        \frac{\partial y_k}{\partial v_k}
    } 
    \textcolor{green}{
        \frac{\partial v_k}{\partial w_{kj}}
    }
\end{equation}


### Backprop

\begin{align}
	\textcolor{red}{
	\frac{\partial E}{\partial y_k}} & = -(d_k - y_k) = -e_k                                           \\
	\textcolor{blue}{
		\frac{\partial y_k}{\partial v_k}
	}                                & = \frac{\partial }{\partial v_k} \varphi (v_k) = \varphi ' (v_k) \\
	\textcolor{green}{
		\frac{\partial v_k}{\partial w_{kj}}
	}                                & = y_j
\end{align}
So
\begin{equation}
  \frac{\partial }{\partial w_{kj}} E(\vw) = 
    \textcolor{red}{
        \frac{\partial E}{\partial y_k}} 
    \textcolor{blue}{
        \frac{\partial y_k}{\partial v_k}
    } 
    \textcolor{green}{
        \frac{\partial v_k}{\partial w_{kj}}
    } = -e_k \varphi ' (v_k) y_j 
\end{equation}


### Backprop



\centering 

![](2018-03-10-13-49-42.png){width=70%} \



- Hence, to update the hidden-output weights

\begin{align}
    w_{kj} (n+1 ) & = w_{kj} (n) - \eta \frac{\partial E}{\partial w_{kj}} \\
    &= w_{kj} (n) + \eta \underbrace{ e_k \varphi ' (v_k)  }_{\delta_k} y_j  \\ 
    &= w_{kj}(n) + \eta \delta_k y_j \quad (\delta \textrm{ rule }) 
\end{align}




### Backprop



\centering 

![](2018-03-10-13-49-42.png){width=70%} \


- For the input-hidden weights,

\begin{equation}
  \frac{\partial }{\partial w_{ji}} E(\vw) = 
    \textcolor{red}{
        \frac{\partial E}{\partial y_j}} 
    \textcolor{blue}{
        \frac{\partial y_j}{\partial v_j}
    } 
    \textcolor{green}{
        \frac{\partial v_j}{\partial w_{ji}}
    }
\end{equation}


\begin{align}
	\textcolor{red}{
		\frac{\partial E}{\partial y_k}
	} & = - \sum_{k}(d_k - y_k) \varphi ' (v_k) w_{kj} \\
	\textcolor{blue}{
		\frac{\partial y_k}{\partial v_k}
	} & = \varphi ' (v_j)                               \\
	\textcolor{green}{
		\frac{\partial v_k}{\partial w_{ji}}
	} & = x_i
\end{align}


### Backprop

\centering 

![](2018-03-10-13-49-42.png){width=70%} \


- So  
\begin{align}
    \frac{\partial }{\partial w_{ji}} E(\vw) &= 
      \textcolor{red}{
          \frac{\partial E}{\partial y_j}} 
      \textcolor{blue}{
          \frac{\partial y_j}{\partial v_j}
      } 
      \textcolor{green}{
          \frac{\partial v_j}{\partial w_{ji}}
      } \\
      &=  - \sum_{k} \underbrace{(d_k - y_k) \varphi ' (v_k) }_{\delta_k} w_{kj} \varphi ' (v_j) x_i  \\
      & = -(\underbrace{\sum_k \delta_k w_{kj}}_{e_j}) \varphi ' (v_j) x_i \\ 
      &= -e_j \varphi ' (v_j )x_i
\end{align}


### Backprop
\centering 

![](2018-03-10-13-49-42.png){width=70%} \



- Hence, to update the input-hidden weights


\begin{align}
    w_{ji} (n+1 ) & = w_{ji} (n) - \eta \frac{\partial E}{\partial w_{ji}} \\
    &= w_{ji} (n) + \eta \underbrace{ e_j \varphi ' (v_j)  }_{\delta_j} x_i  \\ 
    &= w_{ji}(n) + \eta \delta_j x_i   
\end{align}

- The above is called the generalized  rule


### Backprop

![Illustration of the generalized  rule](2018-03-10-15-36-58.png){width=50%}

- The generalized  rule gives a solution to the credit
        (blame) assignment problem




### Hyperbolic tangent function

\centering 

![](2018-03-10-15-37-41.png){width=80%} \ 



### Backprop

\centering 

![](2018-03-10-13-49-42.png){width=70%} \


- For the logistic sigmoid activation, we have

$$\varphi ' ( v ) = a \varphi ( v )[1 -\varphi  ( v )]$$

- hence

\begin{align}
	\delta_ k & = e_k [ay_k (1 - y_k )]                       \\
	          & = ay_k [1 - y_k ][d_ k - y_k ]                \\
	\delta_j  & = ay_ j [1 - y _j ]  \sum_{k} w_{kj} \delta_k
\end{align}




### Backprop

In summary:

\begin{align}
	\frac{\partial }{\partial w_{kj} } E(\vw) & =	-e_k \varphi ' (v_k) y_j \\
	\frac{\partial }{\partial w_{ji}} E(\vw)  & =	-e_j \varphi ' (v_j) x_i
\end{align}

- Backprop learning is local, concerning
    "presynaptic" and "postsynaptic" neurons only
-   How would changing $E(\vw)$ affect the derivation
-   How would changing  $\varphi (\vv)$ affect the derivation


### Backprop illustration

![](2018-03-10-15-46-32.png){width=80%} \ 



### Backprop

- Extension to more hidden layers is straightforward.
    In general we have
$$ \Delta w_ {ji} (n) = \eta \delta _j y_i$$
    - The  rule applies to the output layer and the generalized
 rule applies to hidden layers, layer by layer from the
 output end.
    -   The entire procedure is called backpropagation (error is
 back propagated from the outputs to the inputs)




### MLP design parameters

- Several parameters to choose when designing an
    MLP (best to evaluate empirically)
-   Number of hidden layers
-   Number of units in each hidden layer
-   Activation function
-   Error function





### Universal aproximation theorem

- MLPs can learn to approximate any function, given
    sufficient layers and neurons (an existence proof)
-   At most two hidden layers are sufficient to
    approximate any function. One hidden layer is
    sufficient for any continuous function





### Optimization tricks

- For a given network, local minima of the cost
    function are possible
-   Many tricks exist to try to find better local minima
    - Momentum: mix in gradient from step $n - 1$
    - Weight initialization: small random values
    - Stopping criterion: early stopping
    - Learning rate annealing: start with large $\eta$ , slowly shrink
    - Second order methods: use a separate  $\eta$ for each
 parameter or pair of parameters based on local curvature
    -   Randomization of training example order
    -   Regularization, i.e., terms in $E(w)$ that only depend on $w$

### Learning rate control: momentum



- To ease oscillating weights due to large , some
    inertia (momentum) of weight update is added
$$     \Delta   w_{ji} (n) = \eta \delta_ j y_i + \alpha \Delta w_ {ji} (n - 1),                        0 < \alpha < 1$$

    - In the downhill situation,           $\Delta w_ {ji} (n)\approx   \frac{\eta}{1-\alpha} \delta   _ j y_i$
        - thus accelerating learning by a factor of $1/(1 - \alpha )$
    - In the oscillating situation, it smooths weight change,
        thus stabilizing oscillations




### Weight initialization

- To prevent saturating neurons and break symmetry
    that can stall learning, initial weights (including
    biases) are typically randomized to produce zero
    mean and activation potentials away from saturation
    parts of the activation function
    - For the hyperbolic tangent activation function, avoiding
        saturation can be achieved by initializing weights so that
        the variance equals the reciprocal of the number of
        weights of a neuron





### Stopping criterion
- One could stop after a predetermined number of epochs or
    when the MSE decrease is below a given criterion
-   Early stoping with cross validation: keep part of the
    training set, called validation subset, as a test for
    generalization performance

\centering 

![](2018-03-10-16-00-59.png){width=50%} \ 





### Selecting model parameters: cross validation 


- Must have separate training, validation, and test datasets to avoid over-confidence, over-fitting
-   When lots of data is available, have dedicated sets
-   When data is scarce, use cross-validation
    - Divide the entire training sample into an estimation subset and a validation subset (e.g. 80/20 split)
    -   Rotate through 80/20 splits so that every point is tested on once





### Cross validation illustration

\centering 

![](2018-03-10-16-01-35.png){width=60%} \ 


### MLP applications

- Task: Handwritten zipcode recognition (1989)
- Network description
    - Input: binary pixels for each digit
    - Output: 10 digits
    - Architecture: 4 layers (16x16-12x8x8-12x4x4-30-10)
- Each feature detector encodes only one feature
    within a local input region. Different detectors in
    the same module respond to the same feature at
    different locations through weight sharing. Such a
    layout is called a convolutional net

### Zipcode recognizer architecture

![zipcode recognizer architecture](2018-03-10-16-07-59.png){width=50%}



### Zipcode recognition (cont.)

- Performance: trained on 7300 digits and tested on
    2000 new ones
    - Achieved 1% error on the training set and 5% error on the test set
    -   If allowing rejection (no decision), 1% error on the test set
    -     The task is not easy (see a handwriting example)
- Remark: constraining network design is a way of
    incorporating prior knowledge about a specific
    problem
    - Backprop applies whether or not the network is constrained

### Letter recognition example

- The convolutional net has been subsequently
    applied to a number of pattern recognition tasks
    with state-of-the-art results
    - Handwritten letter recognition
![](2018-03-10-16-09-17.png){width=80%}



### Automatic driving
- ALVINN (automatic land vehicle in a neural network)
![](2018-03-10-16-09-46.png){width=80%} 
    - One hidden layer, one output layer
    - Five hidden nodes, 32 output nodes (steer left - steer right)
    - 960 inputs (30 x 32 image intensity array)
    - 5000 trainable weights
-   Later success of Stanley (won $ 2M DARPA Grand
    Challenge in 2005)

### Other MLP applications

- NETtalk, a speech synthesizer
- GloveTalk, which converts hand gestures to speech
