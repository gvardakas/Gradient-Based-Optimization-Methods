# Gradient-Based-Optimization-Methods
## Description of the project
Ιn the current project we used the cumulative number of recorded cases of covid-19 in Greece from 14 October 2020, until 12 November 2020 in order to compute a 4th degree polynomial model minimizing the corresponding Mean Squared Error. 
To solve the problem, we selected the following optimization methods with derivatives:

1. The **Newton** method with **linear search** under **Wolfe** conditions.
2. The **Newton safe area** method.
3. The **BFGS** method with **linear search** under **Wolfe** conditions.
4. The **steepest descent** method with **linear search** under **Wolfe** conditions.

The proposed model was used to predict the case values of covid-19 in Greece for 5 days after the end of the data, i.e. from 13 to 17 November 2020. The experiments we conducted showed that our polynomial model led to a Mean Squared Error approximately equal to 0.0879 for the first 3 optimization methods, while for the fourth method its mean value was in the region of 42.3598.

## Libraries
1. [Numpy](https://numpy.org/): The fundamental package for scientific computing with Python programming language.
2. [Pandas](https://pandas.pydata.org/): Data analysis and manipulation tool built on top of the Python programming language.
3. [Matplotlib](https://matplotlib.org/): Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python.

