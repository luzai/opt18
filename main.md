# 

### Logistic regression
\begin{center}
\includegraphics[width=.9\textwidth]{2018-04-15-19-48-36.png}
\end{center}


### Logistic regression

-  Assign probability to each outcome
$$P(y=1|x)=\sigma(w^{T}x+b)$$
-  Train to maximize likelihood
$$l(w)=-\Sigma_{n=1}^{N}\sigma(w^{T}x_{n}+b)^{y_{n}}(1-\sigma(w^{T}x_{n}+b))^{(1-y_{n})}$$

-  Linear decision boundary (with $y$ being $0$ or 1)
$$y=I[w^{T}x+b\geq 0]$$

### Support vector machines
\begin{center}
\includegraphics[width=.7\textwidth]{2018-04-15-19-55-03.png}
\end{center}

 



### Support vector machines

-  Enforce a margin of separation $($here, $y\in\{0,1\})$
$$
(2y_{n}-1)w^{T}x_{n}\geq 1,\ \forall n=1\ldots N
$$
-  Train to find the maximum margin
$$
\min\quad \frac{1}{2}||w||^{2}
$$
$$
\textrm{s.t.}\quad (2y_{n}-1)(w^{T}x_{n}+b)\geq 1,\ \forall n=1\ldots N
$$
- Linear decision boundary
$$
\hat{y}=I[w^{T}x+b\geq 0]
$$


### Recap

- Logistic regression focuses on maximizing the
probability of the data. The farther the data lies from the separating hyperplane (on the correct side), the happier LR is.

- An SVM tries to find the separating hyperplane that maximizes the distance of the closest points to the margin (the support vectors). If a point is not a
support vector, it doesn't really matter.

### A different take

-  Remember, in this example $y\in\{0,1\}$

- Another take on the LR decision function uses the
probabilities instead:

$$\hat{y}=\left\{\begin{array}{ll}
1 & \text{if}\ P(y=1|x)\geq P(y=0|x)\\
0 & \text{otherwise}
\end{array}\right.$$
$$
P(y=1|x)\propto\exp(w^{T}x+b)
$$
$$
P(y=0|x)\propto 1
$$


### A different take

- What if we don't care about getting the right
probability, we just want to make the right decision?

- We can express this as a constraint on the likelihood ratio,
$$
\frac{P(y=1|x)}{P(y=0|x)}\underline{>}C
$$
- For some arbitrary constant $c>1.$



### A different take

- Taking the $\log$ of both sides,
$$
\log(P(y=1|x))-\log(P(y=0|x))\geq\log(c)
$$
-  and plugging in the definition of $P,$
$$
w^{T}x+b-0\underline{>}\log(c)
$$
$$
\Rightarrow(w^{T}x+b)\underline{>}\log(c)
$$
- $c$ is arbitrary, so we pick it to satisfy $\log(c)=1$
$$
w^{T}x+b\geq 1
$$


### A different take

- This gives a feasibility problem (specifically the perceptron problem) which may not have a unique solution.

- Instead, put a quadratic penalty on the weights to make the solution unique:
$$
\min\ \frac{1}{2}||w||^{2}
$$
$$\textrm{s.t. }\ (2y_{n}-1)(w^{T}x_{n}+b)\geq 1, \forall n=1\ldots N$$

- This gives us an SVM!

- We derived an SVM by asking LR to make the right _decisions_.

###  The likelihood ratio

- The key to this derivation is the likelihood ratio,
$$
r=\frac{P(y=N|x)}{P(y=0|x)}
$$
$$
\qquad=\frac{\exp(w^{T}x+b)}{1}
$$
$$
\qquad=\exp(w^{T}x+b)
$$
- We can think of a classifier as assigning some cost to $r.$
- Different costs $=$ different classifiers.

###  LR cost

-  Pick  
\begin{align*}
	\textrm{cost}(r)&=\displaystyle \log(1+\frac{1}{r}) \\ 
	&=\log(1+\exp(-(w^{T}x+b)))
\end{align*}

- This is the LR objective (for a positive example)!

### SVM with slack variables

If the data is not linearly separable, we can change the program to:
$$
\min\ \frac{1}{2}||w||^{2}+\sum_{n=1}^{N}\xi_{n}
$$
$$\textrm{s.t. }\ (2y_{n}-1)(w^{T}x_{n}+b)\geq 1-\xi_{n}, \forall n=1\ldots N$$
$$
\xi_{n}\geq 0,\ \forall n=1\ldots N
$$
Now if a point $n$ is misclassified, we incur a cost of $\xi_{n}$, it's distance to the margin.



### SVM with slack variables cost

-  Pick cost 
\begin{align*}
	\textrm{cost}(r) & = \max(0,1-\log(r))\\
	&=\max(0,1-(w^{T}x+b))
\end{align*} 

### LR cost vs SVM cost

Plotted in terms of $r,$
\begin{center}
\includegraphics[width=.9\textwidth]{2018-04-15-20-10-15.png}
\end{center}
### LR cost vs SVM cost

Plotted in terms of $w^{T}x+b,$
\begin{center}
\includegraphics[width=.9\textwidth]{2018-04-15-20-10-37.png}
\end{center}
### Exploiting this connection

- We can now use this connection to derive extensions to each method.

- These might seem obvious (maybe not) and that's usually a good thing.

- The important point though is that they are
_principled_, rather than just hacks. We can trust that they aren't doing anything crazy.

### Kernel trick for LR

- Recall that in it's dual form, we can represent an SVM decision boundary as:
$$
w^{T}\phi(x)+b=\sum_{n=1}\alpha_{n}K(x,\ x_{n})=0
$$
where $\phi(x)$ is an $\infty$-dimensional basis expansion of $x.$

- Plugging this into the LR cost:
$$
\log(1+\exp(-\sum_{n=1}^{N}\alpha_{n}K(x,\ x_{n})))
$$


### Multi-class SVMs

Recall for multi-class LR we have:
$$
P(y=i|x)=\frac{\exp(w_{i}^{T}x+b_{i})}{\sum_{k}\exp(w_{k}^{T}x+b_{k})}
$$

### Multi-class SVMs

Suppose instead we just want the decision rule to satisfy:
$$
\frac{P(y=\dot{b}|x)}{P(y=k|x)}\geq c \quad \forall k\neq i
$$
Taking logs as before, this gives:
$$
w_{i}^{T}x-w_{k}^{T}x\geq 1 \quad \forall k\neq i
$$

### Multi-class SVMs

- This produces the following quadratic program:

$\displaystyle \min \displaystyle \frac{1}{2}||w||^{2}$
$\textrm{s.t. }\ (w_{y_{n}}^{T}x_{n}+b_{y_{n}})-(w_{k}^{T}x_{n}+b_{k})\geq 1, \forall n=1\ldots N, \forall k\neq y_{n}$

### Take-home message

- Logistic regression and support vector machines are closely linked.

- Both can be viewed as taking a probabilistic model and minimizing some cost associated with
misclassification based on the likelihood ratio.

- This lets us analyze these classifiers in a decision theoretic framework.

- It also allows us to extend them in principled ways.



### Which one to use?

- As always, depends on your problem.

- LR gives calibrated probabilities that can be interpreted as confidence in a decision.

- LR gives us an unconstrained, smooth objective.

- LR can be (straightforwardly) used within Bayesian models.

- SVMs don't penalize examples for which the correct decision is made with sufficient confidence. This may be good for
generalization.

- SVMs have a nice dual form, giving sparse solutions when using the kernel trick (better scalability).

