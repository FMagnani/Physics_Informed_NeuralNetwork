# Physics_Informed_NeuralNetwork
Implementation in TF 2.0 of some examples from Maziar Raissi's Physics Informed Neural Networks (PINNs) repository, for personal study.  

The paper *"Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations"* 
by Raissi, Perdikaris and Karniadakis is the reference for this repository.  

I chose two equations: the Schrodinger equation and the Allen-Cahn equation. In the original paper, the first is solved with a **continuous time approach** and the
second with a **discrete time approach**. The software employed by the authors is **Tensorflow 1**.
In this repository, I tried to solve both of these two equations with either the continuous time and the discrete time approach. I also tried to implement the code
using **Tensorflow 2**.  

A quick overview of the results:  
- **Allen-Cahn, continuous time approach**: That's still work-in-progress. Up to now the code is not working well.  
- **Allen-Cahn, discrete time approach**: Raissi implemented this case in Tensorflow 1. In Tensorflow 2, the code works but there are some issues that I cannot solve.  
- **Schrodinger, continuous time approach**: Raissi implemented this case in Tensorflow 1. I could make it in Tensorflow 2 with success.  
- **Schrodinger, discrete time approach**: I could realize the implementation of this case in Tensorflow 1 with good results. In Tensorflow 2, I have the same issues of the Allen-Cahn discrete case.  

In each folder,  
- the file name_plotting.py contains the utilities for the plotting.  
- the file name.py is the main
- the file name_PINN.py contains the implementation of the specific PINN class.
- eventually, some Notebook files are present. They can be used via **Google Colab**.  
  
  
The documentation inside the code is poor yet. I apologize for that.  

