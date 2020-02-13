This is a windows reimplemention of the original package of Spearmint. The original version of Spearmint for Linux can be found at https://github.com/JasperSnoek/spearmint.


Spearmint
---------

Spearmint is a package to perform Bayesian optimization according to the
algorithms outlined in the paper:  

**Practical Bayesian Optimization of Machine Learning Algorithms**  
Jasper Snoek, Hugo Larochelle and Ryan P. Adams  
*Advances in Neural Information Processing Systems*, 2012  

This code is designed to automatically run experiments (thus the code
name 'spearmint') in a manner that iteratively adjusts a number of
parameters so as to minimize some objective in as few runs as
possible.

Spearmint is the result of a collaboration primarily between machine learning researchers at [Harvard University](https://hips.seas.harvard.edu/) and the [University of Toronto](http://learning.cs.toronto.edu/).


Dependencies
------------
This package requires:

* Python 2.7

* [Numpy](http://www.numpy.org/) version 1.6.1+
you can install this package using the command:

      pip install numpy

  or using anaconda command:

      conda install numpy

* [Scipy](http://www.scipy.org/) version 0.9.0+
you can install this package using the command:

      pip install scipy

  or using anaconda command:
      
      conda install scipy    

* [Google Protocol Buffers](https://developers.google.com/protocol-buffers/) (for the fully automated code).
you can install this package using the command:

      pip install protobuf==2.6.0
  
  or using anaconda command:
  
      conda install protobuf=2.6.0
  
	only version 2.6.0 is tested. The newest version of protobuf has some issues with our package
  

Setup
------------
to setup the package, simply run command:

      python setup.py install
      
to test installation, run command:
      
      python main.py --driver=local --method=GPEIOptChooser --method-args=noiseless=1 ./examples/braninpy/config.pb
      

Run Your Experiment
------------
to run your own experiment, put your experiment folder under ./examples, following the structure of any given example
