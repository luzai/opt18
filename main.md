# Definition 

## 

### Perceptrons 

- Architecture: one-layer feedforward net 
    - Without loss of generality, consider a single-neuron perceptron 

\centering
![](2018-03-08-21-55-42.png){ width=85% } \
 

### Definition

$$y=\varphi(\nu)$$
$$\nu=\sum_{i=1}^{m} w_i x_i +b $$

$$\varphi (\nu) = \left\{ \begin{array}{cc}
    1  & \text{ if } \nu \ge 0 \\ 
    -1 & \text{ otherwise } 
    \end{array} \right. $$

Hence a McCulloch-Pitts neuron, but with real-valued inputs

### Pattern recognition 

- With a bipolar output, the perceptron performs a 2-class classification problem, \ie, apples vs. oranges. 
- How do we learn to perform classification? 
- The perceptron is given pairs of input $x_p$ and desired output $d_p$ 
- How can we find $y_p = \varphi (x_p^T w) =d_p, \forall p$

## Decision boundary 

### But first: decision boundary 

- Can we visualize the decision the perceptron would make in classifying every potential point? 
- Yes, it is called the discriminant function 
$$g(x)=x^Tw=\sum_{i=0}^{m} w_i x_i$$
- What is the boundary between the two classes like? 
- This is a linear function of x 


### Decision boundary example 

\centering 

![](2018-03-08-22-28-01.png){ width=50% } \



### Decision boundary 

- For an m-dimensional input space, the decision boundary is an $(m-1)$-dimensional hyperplane perpendicular to $w$. The hyperplane separate  the input space into two halves, with one half having $y=1$, and the other half having $y=-1$ 
    - When $b=0$, the hyperplane goes through the origin. 

\centering 

![](2018-03-08-22-30-37.png){ width=70% } \ 



### Linear separability 

- For a set of input patterns $x_p$, if there exists at least one $w$ that separates $d=1$ patterns from $d=-1$ patterns, then the classification problem is linearly separable. 
    - In other words, there exists a linear discriminant function that produces no classification error. 
    - Examples: AND, OR, XOR 

### Linear separability 

![illustration: **left**: Linear separable, **right**: Not linear separable ](2018-03-08-22-36-07.png){ width=90% }


### Perceptron definition (recap. ) 


$$y=\varphi(\nu)$$
$$\nu=\sum_{i=1}^{m} w_i x_i +b $$

$$\varphi (\nu) = \left\{ \begin{array}{cc}
    1  & \text{ if } \nu \ge 0 \\ 
    -1 & \text{ otherwise } 
    \end{array} \right. $$

Hence a McCulloch-Pitts neuron, but with real-valued inputs

# Learning rule  

## 

### Perceptron learning rule 

- Learn parameters $w$ from examples $(x_p,d_p)$ 
- In an online fashion, \ie, one point at a time 
- Adjust weights as necessary, \ie, when incorrect 
- Adjust weights to be more like $d=1$ points and more like negative $d=-1$ points. 

### Biological analogy 

- Strengthen an active synapse if the postsynaptic neuron fails to fire when it should have fired 
- Weaken an active synapse if the neuron fires when it should not have fired 
- Formulated by Rosenblatt based on biological intuition 

### Quantitatively 

\begin{equation} 
\begin{aligned} 
w(n+1) & = w(n) + \Delta w(n) \\ 
 & = w(n) + \eta [d(n)-y(n)]x(n) \\
\end{aligned}
\end{equation}

- $n$: iteration number, iterating over points in turn 
- $\eta$: step size or learning rate 
- Only updates $w$ when $y(n)$ is incorrect 

### Geometric interpretation 

\centering 

![](2018-03-08-22-45-16.png){ width=55% } \



### Geometric interpretation  

\centering 

![](2018-03-08-22-45-53.png){ width=55% } \



### Geometric interpretation  

\centering 

![](2018-03-08-22-46-08.png){ width=55% } \



### Geometric interpretation  

\centering 

![](2018-03-08-22-46-39.png){ width=55% } \



### Geometric interpretation 

\centering 

![](2018-03-08-22-48-52.png){ width=55% } \



### Geometric interpretation 

- Each weight update moves $w$ closer to $d = 1$ patterns, or away from $d = -1$ patterns.
- Final weight vector in example solves the classification problem
- Is that true in all cases?

### Summary of perceptron learning algorithm 

- Definition: 
    - $w(n)$: (m+1)-by-1 weight vector (including bias) at step n 
- Inputs: 
    - $x(n)$: $n^{th}$ (m+1)-by-1 input vector with first element = 1
    - $d(n)$: $n^{th}$ desired response 
- Initialization: set $w(0)=0$ 
- Repeat until no points are mis-classified 
    - Compute response: $y(n)=\mathrm{sgn}\left[w(n)^T x(n) \right]$ 
    - Update: $w(n+1)=w(n) + \left[d(n) -y(n)  \right]x(n)$
