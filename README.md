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
|Function 1| Two | One| Appears to have stuck in local optimum |
|Function 2| Two | One| There has been progressive improvements over last few weeks|
|Function 3| Three | One| Observed improvements over the time |
|Function 4| Four | One| Observed improvements over the time  |
|Function 5| Four | One| Observed improvements over the time  |
|Function 6| Five | One| Observed improvements over the time |
|Function 7| Six | One| There has been few ups and downs in last few weeks |
|Function 8| Eight | One| There has been few ups and downs in last few weeks | 

Overall, there has been improvements across all function evaluations except for function1 which appears to stuck in local optimum.

## Evaluation Approach
### Gaussian Regressor :

The initial approach was simply an extension of one of the example presented in office hour session using Gaussian Regressor, <span style="color:green; font-style:italic">GaussianProcessRegressor</span> from <span style="color:blue; font-style:italic">sklearn.gaussian_process</span>.
Each function had a separate jupyter notebook, largely similar to each other but a minor difference in acquisition function. In first few weeks I tinkered aroudn the acquisition functions in an attempt to improve the ranking. The legacy code for this iteration is in branch <span style="color:magenta; font-style:italic">**gaussian_regressor**</span>

### BoTorch Expected Improvement :
The approch has changed significantly around week 17 with introduction to BoTorch library. This resulted in code recatoring of those number of separate and difficult to manage jupyter notebooks for each function into a single python program, <span style="color:red; font-weight:bold">bayes_opt.py</span>. In this iteration of the project, saw an experimention with <span style="color:green; font-style:italic">qExpectedImprovement</span> from <span style="color:blue; font-style:italic">botorch.acquisition</span>. Code for this iteration is in branch <span style="color:magenta; font-style:italic">**botorch_EI**</span>

### Monte-Carlo Acquisition Function :
Further to my growing interest in Bayesian Optimisition with BoTorch library and improved ranking, I started experimention with other acquisition functions, especially <span style="color:green; font-style:italic">qExpectedImprovement</span> from <span style="color:blue; font-style:italic">botorch.acquisition.monte_carlo</span>. This is a final third iteration iteration in <span style="color:magenta; font-style:italic">**main**</span> branch.

Further explanation is in the reflection slides.

[Reflection](https://infoscale.github.io/ICBS-AIML-Capstone/reflection.html)

### How to run the bayes optimisation

On terminal prompt run following command
```
 python3 ./bayes_opt.py
 ```

This will generate output as below ....

```
Function 1 results:
Candidates: tensor([[0.5274, 0.0000]])


Function 2 results:
Candidates: tensor([[0.6919, 0.3974]])


Function 3 results:
Candidates: tensor([[0.3751, 0.5810, 0.4198]])


Function 4 results:
Candidates: tensor([[0.3893, 0.3728, 0.4252, 0.4228]])


Function 5 results:
Candidates: tensor([[0.5602, 0.1850, 0.7339, 0.6517]])


Function 6 results:
Candidates: tensor([[0.3640, 0.3210, 0.6556, 0.8008, 0.0966]])


Function 7 results:
Candidates: tensor([[0.1454, 0.2888, 0.5960, 0.2864, 0.3134, 0.6588]])


Function 8 results:
Candidates: tensor([[0.1006, 0.1494, 0.1603, 0.1808, 0.7675, 0.5093, 0.2237, 0.6254]])
```

After receiving results for the new submissions (from above candidates), update the file **bayes_opt.py** with new data points in the functions 1 to 8 to generate next submission set.