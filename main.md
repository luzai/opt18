# 


### Problem statement 

\centering 

![](2018-03-09-15-26-11.png){width=65%} \ 



### Linear regression with one variable 

Given a set of $N$ pairs of data $\{x_i,d_i\}$, approximate $d$ by a linear function of $x$ (regressor), \ie, 
$$d \approx wx +b$$ 
or 
\begin{equation*}
\begin{aligned}
d_i &= y_i + \epsilon_i  = \varphi (wx_i + b ) + \varepsilon \\ 
 & = wx_i + b+\varepsilon 
\end{aligned}
\end{equation*}
where the activation function $\varphi(x) = x$ is a linear function, corresponding to a linear neuron. $y$ is the output of the neuron, and 
$$\varepsilon_i = d_i -y_i$$
is called the (expectational) regression error. 

### Linear regression

- The problem of regression with one variable is how to choose $w$ and $b$ to minimize the regression error. 
- The least squares method aims to minimize the square error 

$$E=\frac{1}{2}\sum_{i=1}^{N} \varepsilon_i^2 = \frac{1}{2} \sum_{i=1}^{N} (d_i -y_i)^2$$ 

### Linear regression 

To minimize the two-variable square function, set 

$$\left\{\begin{array}{cc}
\frac{\partial E}{\partial b} & =0 \\ 
\frac{\partial E}{\partial w} & =0 
\end{array} \right.$$

$$\Rightarrow \left\{\begin{array}{cc}
-\sum_{i}(d_i -wx_i -n) & =0 \\  
-\sum_{i} (d_i -wx_i -b) x_i & =0 
\end{array} \right.$$

### Analytic solution aprroaches 

- Solve one equation for $b$ in terms of $w$ 
    - Substitute into other equation, solve for $w$ 
    - Substitute solution for $w$ back into equation for $b$ 
- Setup system of equations in matrix notation 
    - Solve matrix equation 
- Rewrite problem in matrix form 
    - Compute matrix gradient 
    - Solve for $w$ 

$$\left\{\begin{array}{cc}
-\sum_{i}(d_i -wx_i -n) & =0 \\  
-\sum_{i} (d_i -wx_i -b) x_i & =0 
\end{array} \right.$$

$\Rightarrow \quad$ 
\pause
$b=\frac{\sum_i x_i^2 \sum_i d_i - \sum_i x_i \sum_i x_i d_i}{N \sum_i (x_i - \bar{x} )^2} , \quad$
$w=\frac{\sum_i (x_i - \bar{x}) (d_i - \bar{d}) }{\sum_i (x_i - \bar{x} )^2 }$

, where an $\bar{x}$ indicates the mean 

### Linear regression in matrix notation

Let $\mX = [\vx_1 ,\vx_2 ,\vx_3 ,\dots ,\vx_N]^T$, then the model predictions are $\vy=\mX \vw$. 
And the mean square error can be written as 
$$E(\vw)=\|\vd - \vy\|^2 = \|\vd -\mX \vw\|^2$$ 

To find the optimal $\vw$, set the gradient of the error \wrt $\vw$ equal to 0 and solve for $\vw$. 

$$\partial E(\vw) / \partial \vw=0$$

<!-- [^1]: ref to The Matrix Cookbook -->

### Linear regression in matrix notation

\begin{equation*}
\begin{aligned}
\frac{\partial}{\partial \vw} E(\vw) & = \frac{\partial}{\partial \vw} \| \vd - \mX \vw \|^2 \\  
&= \frac{\partial}{\partial \vw} (\vd -\mX \vw)^T(\vd -\mX \vw) \\ 
&= \frac{\partial}{\partial \vw} \vd^T \vd -2 \vw^T \mX^T \vd + \vw^T \mX^T \mX \vw \\ 
&= -2 \mX^T \vd -2 \mX^T \mX \vw 
\end{aligned}
\end{equation*}

$$\Rightarrow \vw = (\mX ^T \mX)^{-1} \mX^T d$$

### Finding optimal parameters via search

- Often there is no closed form solution for $\frac{\partial}{\partial \vw} E(\vw)=0$
- We can still use the gradient in a numerical solution
- We will still use the same example to permit comparison
- For simplicity’s sake, set $b = 0$

$$E(w) = 1/2 \sum_{i=1}^{N} (d_i - wx_i)^2$$

, where $E(w)$ is called cost function. 

### Cost function 

\centering 

![](2018-03-09-22-15-39.png){width=50%} \


Question: how can we update $w$ from $w_0$ to minimize $E$? 

### Gradient and directional derivatives 

Consider a two-variable function $f(x,y)$. Its gradient at the point $(x_0 ,y_0)^T$ is defined as 

\begin{equation*}
    \begin{aligned}
        \nabla f & = \left. (\partial f(x,y) / \partial x , \partial f(x,y) / \partial y)^T \right|_{x=x_0,y=y_0}   \\ 
        & = f_x(x_0,y_0) \vu_x + f_y(x_0,y_0) \vu_y 
    \end{aligned}
\end{equation*}

, where $\vu_x$ and $\vu_y$ are unit vectors in the x and y directions, and $f_x=\partial f / \partial x$ and $f_y = \partial f / \partial y$ 

### Gradient and directional derivatives 

At any given direction, $\vu = \alpha \vu_x + b \vu_y$, with $\sqrt{a^2+b^2}=1$, the directional derivative at $(x_0, y_0 )^T$ along the unit vector $\vu$ is 

\begin{equation*}
    \begin{aligned}
        D_u f_x(x_0,y_0) &= \lim_{h \rightarrow 0} \frac{f(x_0 + ha, y_0+hb)-f(x_0,y_0) }{h} \\  
        &= \lim_{h \rightarrow 0} \frac{[f(x_0+ha , y_0 +hb)-f(x_0,y_0+hb)]+[f(x_0,y_0+hb)-f(x_0,y_0)]}{h} \\
        &= af_x(x_0,y_0)+bf_y(x_0,y_0) \\
        &= \nabla f(x_0,y_0)^T \vu
    \end{aligned}
\end{equation*}

Which direction has the greatest slope? The gradient! Because of the dot product. 

### Gradient and directional derivatives 

Example: $f(x,y)=5/2 x^2 -3xy + 5/2 y^2 +2x +2y$

\centering 

![](2018-03-09-22-31-32.png){width=70%} \


### Gradient and directional derivatives 

Example: $f(x,y)=5/2 x^2 -3xy + 5/2 y^2 +2x +2y$

\centering 

![](2018-03-09-22-34-05.png){width=70%} \




### Gradient and directional derivatives (cont.)
- The level curves of a function $f(x,y$ are curves such that
    $f(x,y)=k$
-   Thus, the directional derivative along a level curve is 0
               $$D_\vd = \nabla f(x_0,y_0) ^T \vu = 0 $$
- And the gradient vector is perpendicular to the level curve





### Gradient and directional derivatives (cont.)
- The gradient of a cost function is a vector with the
    dimension of w that points to the direction of maximum $E$
    increase and with a magnitude equal to the slope of the
    tangent of the cost function along that direction
     - Can the slope be negative?





### Gradient illustration

\centering 

![](2018-03-10-09-23-28.png){width=80%} \
                              


###                       Gradient descent
- Minimize the cost function via gradient (steepest) descent 
    a case of hill-climbing
              $$w( n + 1) = w( n ) − \eta \nabla E ( n )$$

    - $n$: iteration number
    - $\eta$: learning rate
    - See previous figure





###                   Gradient descent (cont.)
- For the mean-square-error cost function and linear neurons
\begin{equation*}
    \begin{aligned}
        E ( n ) & = \frac{1}{2} e^2 ( n ) = \frac{1}{2} [d ( n ) − y ( n )]^2  \\
        &= \frac{1}{2} [d ( n ) − w( n ) x ( n )]^2  \\
        \nabla E(n) &= \frac{\partial E}{\partial w(n)  }  = \frac{\partial e^2 (n)}{2 \partial w(n) } \\
        &= -e(n)x(n) 
    \end{aligned}
\end{equation*}
       


###                   Gradient descent (cont.)
- Hence
\begin{eqnarray*}
    w( n + 1) &=& w( n ) +\eta  e( n ) x ( n ) \\ 
               &     =& w( n ) +\eta  [d ( n )  y ( n )] x ( n )
\end{eqnarray*}
          

- This is the least-mean-square (LMS) algorithm, or the Widrow-Hoff
         rule





###                   Stochastic gradient descent

- If the cost function is of the form
                                    
$$E(w)=\sum_{n=1}^{N}E_n(w) $$
                            
- Then one gradient descent step requires computing
$$\Delta = \frac{\partial}{\partial w} E(w) =\sum_{n=1}^{N} \frac{\partial}{\partial w}E_n(w) $$                            
- Which means computing $E(w)$ or its gradient for
    every data point
-   Many steps may be required to reach an optimum

###                   Stochastic gradient descent


- It is generally much more computationally efficient
    to use
    $$\Delta =  \sum_{n=n_i}^{n_i+n_b-1} \frac{\partial}{\partial w}E_n(w) $$   
- For small values of $n_b$ 
- This update rule may converge in many fewer
    passes through the data (epochs)




###        Stochastic gradient descent example

\centering 

![](2018-03-10-10-00-53.png){width=80%} \



### Stochastic gradient descent error functions

\centering 


![](2018-03-10-10-01-05.png){width=80%} \



###       Stochastic gradient descent gradients

\centering 

![](2018-03-10-10-01-13.png){width=80%} \




###      Stochastic gradient descent animation

\centering 

![](2018-03-10-10-01-21.png){width=80%} \




###                   Gradient descent animation

\centering 

![](2018-03-10-10-01-33.png){width=80%} \



###                    Multi-variable LMS
- The analysis for the one-variable case extends to the multi-
    variable case

$$ E ( n ) = 1/2 [d ( n )  \vw^ T ( n )\vx( n )]^2 $$

$$  \nabla  E( w ) = \left(  \frac{\partial E}{\partial w_0}  ,  \frac{\partial E}{\partial w_1}  ,...,   \frac{\partial E}{\partial w_m} \right)^T $$

where $w_0= b$ (bias) and $x_0 = 1$, as done for perceptron learning





###                   Multi-variable LMS (cont.)
- The LMS algorithm

\begin{eqnarray*}
    \vw ( n + 1)& =& \vw ( n )-\eta \nabla  \mE( n ) \\ 
    &= &\vw ( n ) + \eta e( n )\vx( n )\\
    &= &\vw ( n ) + \eta [d ( n )  y ( n )]\vx( n ) 
\end{eqnarray*}
    





###                   LMS algorithm remarks

- The LMS rule is exactly the same equation as the
    perceptron learning rule
-   Perceptron learning is for nonlinear (M-P) neurons,
    whereas LMS learning is for linear neurons.
     - \ie , perceptron learning is for classification and LMS is
         for function approximation
- LMS should be less sensitive to noise in the input
    data than perceptrons
     - On the other hand, LMS learning converges slowly
-   Newtons method changes weights in the direction
    of the minimum $E(w)$ and leads to fast convergence.
     - But it is not online and is computationally expensive


###                   Stability of adaptation

\begincols{} 

\column{0.6\textwidth} 

![](2018-03-10-10-13-18.png){width=100%} \



\column{0.4 \textwidth}

- When $\eta$ is too small,
learning converges slowly
- When $\eta$ is too large, learning

\stopcols



###                   Learning rate annealing
- Basic idea: start with a large rate but gradually decrease it
- Stochastic approximation 
$$\eta(n) = c/n$$
               
     $c$ is a positive parameter





###             Learning rate annealing (cont.)
- Search-then-converge
$$\eta(n) = \frac{\eta_0}{1+(n/\tau)}$$
     $\eta_0$ and $\tau$ are positive parameters


     When n is small compared to $\tau$ , learning rate is approximately constant
     When n is large compared to $\tau$ , learning rule schedule roughly follows
     stochastic approximation





###                   Rate annealing illustration

\centering  

![](2018-03-10-10-18-17.png){width=70%} \





###                       Nonlinear neurons
- To extend the LMS algorithm to nonlinear neurons, consider
    differentiable activation function  at iteration n
    \begin{eqnarray*}
           E (n) &=& 1/2 [d (n)  y (n)]^2 \\  
                &=& 1/2 \left[ d (n)  - \varphi  \sum_{j}  w_j x_j (n) \right] ^2  
    \end{eqnarray*}



###                    Nonlinear neurons (cont.)
- By chain rule of differentiation
\begin{eqnarray*}
\frac{\partial E}{\partial w_j} &=&
 \frac{\partial E}{\partial y}\frac{\partial y}{\partial v}\frac{\partial v}{\partial w_j} \\ 
&=& - [d (n) - y (n)]\varphi' (v(n) )x_ j (n) \\ 
&=& - e(n) \varphi' (v(n) ) x_ j (n)
\end{eqnarray*}





###                   Nonlinear neurons (cont.)
- Gradient descent gives
       \begin{eqnarray*}
       w _j (n + 1) &=& w_ j (n) +\eta e(n)\varphi' (v(n)) x _j (n) \\ 
                         &=& w_ j (n) +\eta \delta (n) x_ j (n) 
       \end{eqnarray*} 
     - The above is called the delta ($\delta$) rule
-   If we choose a logistic sigmoid for 
$$                    \varphi (v) = \frac{1}{  1+ exp( - av )} $$
     then

$$                   \varphi '      ( v ) = a \varphi ( v )[1-\varphi   ( v )]   $$


###                   Role of activation function
                                    
\centering 

![](2018-03-10-10-41-52.png){width=100%} \ 


The role of : weight update is most sensitive when v is near zero
