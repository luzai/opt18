#

###      Motivation

Multilayer networks are more powerful than single layer nets, \eg, XOR problem


<!--
### Power of nonlinearity

 For linear neurons, a multilayer net is equivalent to
    a single-layer net. This is not the case for nonlinear
    neurons
      Why?





### MLP architecture





### Multi-layer perceptron

 Think of an MLP as a complicated, non-linear
    function of its input parametrized by :
                        =  ; 
   Note that Multi-layer perceptron is a bit of a
    misnomer because they use a continuous activation
    function





###        MLP Training

 Given a set of training data  ,  can we adjust
     so that the network is optimal?
   Optimal with respect to what criterion?
      Must define error criterion btwn  = ( ; ) and 
      We will use the mean square error for now, but others are
        possible (and often preferable)
 Goal find  that minimizes
                                                               2
                            1
        =    =      ; 
                            2
                                 



###    Backpropagation

 Because  () is still a complication non-linear
    function, we will optimize it using gradient descent
   Because of the structure of MLPs, we can compute
    the gradient of  () very efficiently using the
    backpropagation algorithm
   Backpropagation computes the gradient of each
    layer recursively based on subsequent layers
   Because this is  () and not  (), we will be
    using stochastic gradient descent


###                            Notation

 Notation for one hidden layer (drop p for now)
                         wji   vj           yj w     vk          yk
             xi                             kj
                                                                  E




                 Input              Hidden            Output


###                    Notation

 Notation for one hidden layer (drop p for now)
                  wji   vj          yj w      vk          yk
             xi                     kj
                                                           E

                   =       
                                                                    
                            1                                  2
                    () =    
                            2
                                         
 Keep in mind during the derivation:
      How would changing  () affect the derivation?
      How would changing   affect the derivation?

###               Backprop
                                                   wji vj          yj wkj vk          yk
                                             xi                                         E
                            1                          2
                       =    
                            2
                                            vk
                                                                    2
                  1
                 =       
                  2                                       
                      
                                              yk

 Then, to adjust the hidden-output weights
                                     
                              =
                              


###              Backprop
                                                 wji vj          yj wkj vk          yk
                                            xi                                        E
                   
                          =     = 
                  
                       
                          =         =   
                   
                   
                           = 
                  w
So
                         
                  =                        =     
                  



###            Backprop
                                                wji vj           yj wkj vk          yk
                                        xi                                            E
 Hence, to update the hidden-output weights
                                          E
            wkj ( n + 1) = wkj ( n )  
                                         wkj

                           = wkj (n) + ek (vk ) y j
                                                   
                            = wkj (n) +  k y j                         ( rule)





###                      Backprop
                                                        wji vj          yj wkj vk          yk
                                             xi                                              E
 For the input-hidden weights,
                                    
                              =
                              
                                                         vk
                                                                               2
               1
               =               
           2                                                
                               
                                                         yk
                 =         
                              
                                        
                          =                      = 
                                       

###               Backprop
                                                     wji vj          yj wkj vk          yk
                                              xi                                          E
 So
                                     
                               =
                               
             =             
                                 

                    =        
                              
                                   ej

                            =     


###     Backprop
                                      wji vj           yj wkj vk          yk
                                 xi                                         E
 Hence, to update the input-hidden weights
                                  E
     w ji (n + 1) = w ji (n)  
                                 w ji
                     = w ji (n) + e j (v j ) xi
                                         
                      = w ji (n) +  j xi
 The above is called the generalized  rule


###           Backprop
                                              wji vj          yj wkj vk          yk
                                        xi                                         E
 Illustration of the generalized  rule,
                                             1
                          j
                 xi        j        k        k




      The generalized  rule gives a solution to the credit
        (blame) assignment problem




### Hyperbolic tangent function





###             Backprop
                                                    wji vj          yj wkj vk          yk
                                            xi                                           E
 For the logistic sigmoid activation, we have
                  ( v ) = a ( v )[1   ( v )]

      hence
                    k = ek [ayk (1  yk )]
                        = ayk [1  yk ][d k  yk ]

                    j = ay j [1  y j ] wkj k
                                           k




###            Backprop
                                               wji vj          yj wkj vk          yk
                                          xi                                        E
In summary:
                    
                             =     
                 
                     
                             =     
                  
 Backprop learning is local, concerning
    presynaptic and postsynaptic neurons only
   How would changing () affect the derivation?
   How would changing   affect the derivation?


### Backprop illustration





###            Backprop

 Extension to more hidden layers is straightforward.
    In general we have
                       w ji (n) =  j yi
      The  rule applies to the output layer and the generalized
          rule applies to hidden layers, layer by layer from the
         output end.
        The entire procedure is called backpropagation (error is
         back propagated from the outputs to the inputs)




### MLP design parameters

 Several parameters to choose when designing an
    MLP (best to evaluate empirically)
   Number of hidden layers
   Number of units in each hidden layer
   Activation function
   Error function





             Universal approximation theorem

 MLPs can learn to approximate any function, given
    sufficient layers and neurons (an existence proof)
   At most two hidden layers are sufficient to
    approximate any function. One hidden layer is
    sufficient for any continuous function





###      Optimization tricks

 For a given network, local minima of the cost
    function are possible
   Many tricks exist to try to find better local minima
      Momentum: mix in gradient from step   1
      Weight initialization: small random values
      Stopping criterion: early stopping
      Learning rate annealing: start with large , slowly shrink
      Second order methods: use a separate  for each
         parameter or pair of parameters based on local curvature
        Randomization of training example order
        Regularization, i.e., terms in E(w) that only depend on w

             Learning rate control: momentum

 To ease oscillating weights due to large , some
    inertia (momentum) of weight update is added

     w ji (n) =  j yi + w ji (n  1),                        0 < <1
                                                         
      In the downhill situation,           w ji (n)       j yi
                                                        1
           thus accelerating learning by a factor of 1/(1  )
      In the oscillating situation, it smooths weight change,
        thus stabilizing oscillations




###    Weight initialization

 To prevent saturating neurons and break symmetry
    that can stall learning, initial weights (including
    biases) are typically randomized to produce zero
    mean and activation potentials away from saturation
    parts of the activation function
      For the hyperbolic tangent activation function, avoiding
        saturation can be achieved by initializing weights so that
        the variance equals the reciprocal of the number of
        weights of a neuron





###     Stopping criterion
 One could stop after a predetermined number of epochs or
    when the MSE decrease is below a given criterion
   Early stopping with cross validation: keep part of the
    training set, called validation subset, as a test for
    generalization performance





 Selecting model parameters: (cross-)validation

 Must have separate training, validation, and test
    datasets to avoid over-confidence, over-fitting
   When lots of data is available, have dedicated sets
   When data is scarce, use cross-validation
      Divide the entire training sample into an estimation
         subset and a validation subset (e.g. 80/20 split)
        Rotate through 80/20 splits so that every point is tested
         on once





### Cross validation illustration





###  MLP applications

 Task: Handwritten zipcode recognition (1989)
 Network description
    Input: binary pixels for each digit
    Output: 10 digits
    Architecture: 4 layers (16x1612x8x812x4x43010)
 Each feature detector encodes only one feature
  within a local input region. Different detectors in
  the same module respond to the same feature at
  different locations through weight sharing. Such a
  layout is called a convolutional net


### Zipcode recognizer architecture





### Zipcode recognition (cont.)

 Performance: trained on 7300 digits and tested on
    2000 new ones
      Achieved 1% error on the training set and 5% error on
         the test set
        If allowing rejection (no decision), 1% error on the test
         set
        The task is not easy (see a handwriting example)
 Remark: constraining network design is a way of
  incorporating prior knowledge about a specific
  problem
    Backprop applies whether or not the network is
         constrained

### Letter recognition example

 The convolutional net has been subsequently
    applied to a number of pattern recognition tasks
    with state-of-the-art results
      Handwritten letter recognition





###        Automatic driving
 ALVINN (automatic land vehicle in a neural network)




      One hidden layer, one output layer
      Five hidden nodes, 32 output nodes (steer left  steer right)
      960 inputs (30 x 32 image intensity array)
      5000 trainable weights
   Later success of Stanley (won $2M DARPA Grand
    Challenge in 2005)

### Other MLP applications

 NETtalk, a speech synthesizer
 GloveTalk, which converts hand gestures to speech
-->