# 

## 

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