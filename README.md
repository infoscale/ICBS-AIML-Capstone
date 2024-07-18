# Imperial College Business School
## Professional Certificate in Machine Learning and Artificial Intelligence
## Capstone - Bayesian Optimisation

Bayesian optimisation is a blackbox optimisation technique to solve an objective function that are expensive to evaluate. 
The process iteratively evaluate the function at s[ecific points to find the optimal solution.

Bayesian optimisation uses Gausian process which provides a probabilistic way to model the functions behavious. each evaluation informs the next query point.

Acquisition functions guides the search by balancing exploration and exploitation.

## Capstone project
This part of a capstone project involves evaluation of eight functions in total with different dimensions.

|Function|Input Dimensions|Output Dimension| Leaderboard |
|--------|----------------|----------------|------------|
|Function 1| Two | One| Rank 44, Appears to have stuck in local minimum |
|Function 2| Two | One| Rank 13, there has been progressive improvements over last few weeks|
|Function 3| Three | One| Rank 28, observed improvements over the time |
|Function 4| Four | One| Rank 9, observed improvements over the time  |
|Function 5| Four | One| Rank 3, observed improvements over the time  |
|Function 6| Five | One| Rank 6, observed improvements over the time |
|Function 7| Six | One| Rank 11, there has been few ups and downs in last few weeks |
|Function 8| Eight | One| Rank 12, there has been few ups and downs in last few weeks | 

Overall, there has been improvements across all function evaluations except for function1 which appears to stuck in local minimum.

## Evaluation Approach
### Gaussian Regressor :

The initial approach was simply an extension of one of the example presented in office hour session using Gaussian Regressor, <span style="color:green; font-style:italic">GaussianProcessRegressor</span> from <span style="color:blue; font-style:italic">sklearn.gaussian_process</span>.
Each function had a separate jupyter notebook, largely similar to each other but a minor difference in acquisition function. In first few weeks I tinkered aroudn the acquisition functions in an attempt to improve the ranking. The legacy code for this iteration is in branch <span style="color:magenta; font-style:italic">gaussian_regressor</span>

### BoTorch Expected Improvement :
The approch has changed significantly around week 17 with introduction to BoTorch library. This resulted in code recatoring of those number of separate and difficult to manage jupyter notebooks for each function into a single python program, <span style="color:red; font-weight:bold">bayes_opt.py</span>. In this iteration of the project, saw an experimention with <span style="color:green; font-style:italic">qExpectedImprovement</span> from <span style="color:blue; font-style:italic">botorch.acquisition</span>. Code for this iteration is in branch <span style="color:magenta; font-style:italic">botorch_EI</span>

### Monte-Carlo Acquisition Function :
Further to my growing interest in Bayesian Optimisition with BoTorch library and improved ranking, I started experimention with other acquisition functions, especially <span style="color:green; font-style:italic">qExpectedImprovement</span> from <span style="color:blue; font-style:italic">botorch.acquisition.monte_carlo</span>. This is a final third iteration iteration in <span style="color:magenta; font-style:italic">main</span> branch.

Further explanation is in the reflection slides.

[Reflection](https://infoscale.github.io/ICBS-AIML-Capstone/reflection.html)
