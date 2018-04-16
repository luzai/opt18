# 


### Linear Classifier and the Perceptron Algorithm
	
- $f(x)=\sigma(w^T{x}+b)$ 
- Note: vector or matrix can be judged by context, e.g., here $w$ and $x$ is vector.
- $\sigma$: Sigmoid function $\sigma(x)=\frac{1}{1+e^{-x}}$

- The connection to logistic regression:
	- Assume binomial distribution with parameter $\hat{p}$
	- Assume the logit transform is linear:
$$\log\frac{\hat{p}}{1-\hat{p}}=w^{{T}}x+b$$
$$\Rightarrow\ \hat{p}=\sigma(f(x))$$
	 
###  Maximum Log-Likelihood

- MLE of the binomial likelihood:

$$\sum_{i=1} ^{n} y_i^* \log \hat{p} + (1-{y}_{i}^{*})\log(1-\hat{p})$$

where ${y}_{i}^{*}\in  \{0,1\} =\frac{1+y_{i}}{2}$

$$\log\hat{p}=-\log(1+e^{-f(x)})$$
$$\log(1-\hat{p})=-\log(1+e^{f(x)})$$
$${y}_{{i}}^{*}\log\hat{p}+(1-{y}_{i}^{*})\log(1-\hat{p})=-\log(1+e^{-yf(x)})$$

### Gradient descent optimization

- Optimize $w, b$ with gradient descent

$$\min_{w,b}\sum_{i}\log(1+e^{-y_{i}(w^{{T}}x_{i}+b)})$$

$$\nabla w=\sum_{i}\frac{-{y}_{i}e^{-y_{i}(w^{{T}}x_{i}+b)}}{1+e^{-y_{i}(w^{{T}}x_{i}+b)}}x_{i}=\sum_{i}-{y}_{i}^{*}(1-\hat{p}(x_{i}))-(1-{y}_{i}^{*})\hat{p}(x_{i})x_{i}$$

$$\nabla b=\sum_{i}\frac{-{y}_{i}e^{-y_{i}(w^{{T}}x_{i}+b)}}{1+e^{-y_{i}(w^{{T}}x_{i}+b)}}$$



### XOR problem and linear classifier

- 4 points: ${X}=[(-1,-1),\ (-1,1),\ (1,-1),\ (1,1)]$

- $Y=[-1, 1, 1, -1]$

- Try using binomial $\log$-likelihood loss:
$$
\min_{w,b}\sum_{i}\log(1+e^{-y_{i}(w^{{T}}x_{i}+ b)})
$$
- Gradient:
$$\nabla w=\sum_{i}\frac{-{y}_{i}e^{-y_{i}(w^{{T}}x_{i}+b)}}{1+e^{-{y}_i(w^{{T}}x_{i}+b)}}x_{i}$$
$$\nabla b=\sum_{i}\frac{-{y}_{i}e^{-y_{i}(w^{{T}}x_{i}+b)}}{1+e^{-{y}_i(w^{{T}}x_{i}+b)}}$$

- Try $w=0, b=0$,  what do you see?

### With 1 hidden layer

\begin{columns}
	\begincols{.5\textwidth}
		\begin{itemize}
			\item A hidden layer makes a nonlinear classifier
			$$
			f(x)=w^{{T}}g(W^{{T}}x+c)+b
			$$
			\item  ${g}$ needs to be nonlinear
			\item Sigmoid: $\sigma(x)=1/(1+e^{-x})$
			\item RELU: $g({x})=\max(0,{x})$
		\end{itemize}
	\stopcols
	\begincols{.5\textwidth}
	\begin{center}
		\includegraphics[width=\textwidth]{2018-04-15-13-08-28.png}
	\end{center}
	\stopcols
		
\end{columns}


### Taking gradient
$$
\min_{W,w}E(f)=\sum_{i}L(f(x_{i}),y_{i})
$$
$$
f(x)=w^{T}g(W^{T}x+c)+b
$$

- What is $\frac{\partial E}{\partial W}$ ?

- Consider chain rule: $\frac{dz}{dx}=\frac{dz}{dy}\frac{dy}{dx}$

### Note: Vectorized Computations
 
- The computations performed by a network. 
$$z_i = \sum_{i} W_{ij} x_j$$
$$h_i=\sigma(z_i)$$
$$y=\sum_i v_i h_i$$
- Write them in terms of matrix and vector operations. Note: judge vector or matrix by context 
$$z=Wx$$
$$h=\sigma(z)$$
$$y=v^T h$$
- $\sigma(v)$ denotes the logistic sigma function applied elementwise to a vector $v$. $W$ is a matrix where the $(i,j)$ entry is the weight from visible unit $j$ to hidden unit $i$. 

 

### Backpropagation

- Save the gradients and the gradient products that have already been computed to avoid computing multiple times
- In a multiple layer network(Ignore constant terms)    
\begin{center}
	\includegraphics[width=\textwidth]{2018-04-15-13-09-15.png}
\end{center}

\end{columns}

### Backpropagation

- Save the gradients and the gradient products that have already been computed to avoid computing multiple times
- In a multiple layer network(Ignore constant terms)    
$$f(x)=w_{n}^{T}
g\bb{  
	W_{n-1}^{T}
	g\bb{
		W_{n-2}^{T}g
		\bb{
			W_{1}^{T}g\bb{x}
		}
	}
}$$
		$$
		\frac{\partial E}{\partial W_{k}}=\frac{\partial E}{\partial f_{k}}g(f_{k-1}(x))
		=\frac{\partial E}{\partial f_{k+1}}\frac{\partial f_{k+1}}{\partial f_{k}}g(f_{k-1}(x))
		$$
- Define: $f_{k}(x)=w_{k}^{T}g(f_{k-1}(x)),  f_{0}(x)=x$

### Modules 

\begin{itemize}
	\tightlist
	\item Each layer can be seen as a module
	\item Given input, return
	      	\begin{itemize}
	      		\tightlist
	      		\item  Output $f_{a}(x)$
	      		\item   Network gradient $\frac{\partial f_{a}}{\partial x}$
	      		\item  Gradient of module parameters $\frac{\partial f_{a}}{\partial w_a}$
	      	\end{itemize}
	      	\begin{center}
				\includegraphics[width=.45\textwidth]{2018-04-15-13-09-43.png}
			\end{center}
\end{itemize}
 

### Modules
\begin{itemize}
	\tightlist
	\item Each layer can be seen as a module
	\item Given input, return
	\begin{itemize}
		\tightlist
		\item  Output $f_{a}(x)$
		\item   Network gradient $\frac{\partial f_{a}}{\partial x}$
		\item  Gradient of module parameters $\frac{\partial f_{a}}{\partial w_a}$
	\end{itemize}
\end{itemize}
 
- During backprop, propagate/update
	- Backpropagated gradient $\frac{\partial E}{\partial f_{a}}$
$$\frac{\partial E}{\partial W_{k}}=
\frac{\partial E}{\partial f_{k}} g(f_{k-1}( x))=
\red{\frac{\partial E}{\partial f_{k + 1} }}
\blue{\pp{f_{k+1}}{f_k}}
\green{g(f_{k-1}(x))}$$ 
	-  \red{Backprop signal} \blue{Network Gradient} \green{gradient of  parameters}
	- Note: $\frac{\partial E}{\partial f_{k}}=\pp{E}{f_{k+1}} \pp{f_{k+1}}{{f_k}}$


### Multiple Inputs and Multiple Outputs

$$\frac{\partial E}{\partial f_{k-1}}=\pp{E}{f_{k+1}}\pp{f_{k+1}}{f_{k_1}}\pp{f_{k_1}}{f_{k-1}}+\pp{E}{f_{k+1}} \pp{f_{k+1}}{f_{k_2}}\pp{f_{k_2}}{f_{k-1}}$$

\begin{center}
	\includegraphics[width=.4\textwidth]{2018-04-15-13-10-08.png}
\end{center}

###  Different DAG structures

- The backpropation algorithm would work for any DAGs
- So one can imagine different architectures than the plain layerwise one

\begin{center}
	\includegraphics[width=.9\textwidth]{2018-04-14-00-19-13.png}
\end{center}

### Loss functions

- Regression:
	- Least squares $L(f)=(f(x)-y)^{2}$
	- L1 loss $L(f)=|f(x)-y|$
	- Huber loss $$L({f})=\begin{cases} \frac{1}{2} (f(x)-y)^2 & , |f(x)-y| \le \delta \\ \delta (|f(x)-y| -\frac{1}{2} \delta ) & \textrm{, otherwise} \end{cases}$$

\begin{center}
	\includegraphics[width=.42\textwidth]{2018-04-15-13-10-18.png}
\end{center}


### Loss functions

- Regression:
	- Least squares $L(f)=(f(x)-y)^{2}$
	- L1 loss $L(f)=|f(x)-y|$
	- Huber loss $$L({f})=\begin{cases} \frac{1}{2} (f(x)-y)^2 & , |f(x)-y| \le \delta \\ \delta (|f(x)-y| -\frac{1}{2} \delta ) & \textrm{, otherwise} \end{cases}$$
- Binary Classification
	- Hinge loss $L(f)=\max(1-yf(x),  0)$
	- Binomial log-likelihood $L(f)=\ln(1+\exp(-2yf(x))$
	- Cross-entropy $L(f)=-y^{*}\ln \sigma(f)-(1-y^{*})\ln(1- \sigma(f))$ ,
		- $y^{*}=(y+1)/2$


### Multi-class: Softmax layer

- Multi-class logistic loss function

$$P(y=j|x)=\frac{e^{x^Tw_j}}{\sum_{k=1}^{K}e^{x^Tw_k}}$$

- Log-likelihood:
- Loss function is minus $\log$-likelihood
$$
-\log P(y=j|x)=-x^{T}w_{\dot{j}}+\log\sum_{k}e^{x^{T}w_{k}}
$$

### Subgradients

- What if the function is non-differentiable?
- Subgradients:
	- For convex $f(x)$ at $x_{0}$:  
	- If for any $y$
	$$f(y)\geq f(x)+g^T(y-x)$$
	- $g$ is called a subgradient
- Subdifferential: $\partial f$: set of all subgradients
- Optimality condition: $0\in\partial f$
  
\begin{center}
	\includegraphics[width=.375\textwidth]{2018-04-15-13-10-32.png}
\end{center}

### The RELU unit

- $f(x)=\max(x,0)$
- Convex
- Non-differentiable
- Subgradient: $\frac{\partial f}{\partial x}=\begin{cases} 1 &,x>0 \\ [0,1]&,x=0 \\ 0&,x<0  \end{cases}$


\begin{center}
	\includegraphics[width=.5\textwidth]{2018-04-15-13-10-48.png}
\end{center}

### Subgradient descent

- Similar to gradient descent
$$x^{(k+1)}=x^{(k)}-\alpha_k g^{(k)}$$
- Step size rules:
	- Constant step size: $\alpha_k = \alpha$
	- Square summable: $\alpha_k\ge 0, \sum_{k=1}^{\infty} \alpha_k^2 < \infty, \sum_{k=1}^{\infty}\alpha_k =\infty$ 
	- Usually, a large constant that drops slowly after a long while . e.g. $\frac{100}{100+k}$

### Universal Approximation Theorems
- Many universal approximation theorems proved in the $90s$
- Simple statement: for every continuous function, there exist a function that can be approximated by a 1-hidden layer neural network with arbitrarily high precision

\begin{center}
	\includegraphics[width=\textwidth]{2018-04-15-13-11-16.png}
\end{center}

### Universal Approximation Theorems

- The approximation does not need many units if the function is kinda nice. Let
$$C_{f}=\int_{R_{d}}||\omega|||\tilde{f}(\omega)|d\omega$$
- Then for a 1-hidden layer neural network with $n$ hidden nodes, we have for a finite ball with radius $r$, 
$$
\int_{B_{r}}(f(x)-f_{n}(x))^{2}d \mu(x)\leq\frac{4r^{2}C_{f}^{2}}{n}
$$





