import torch
import warnings
import numpy as np
from botorch.models import SingleTaskGP
from botorch import fit_gpytorch_mll
from botorch.acquisition import qExpectedImprovement
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf


def function_1():
    X = np.load("./function_1/initial_inputs.npy")
    Y = np.load("./function_1/initial_outputs.npy")
    # Add some more points to the training data
    X = np.append(
        X,
        np.array(
            [
                [0.0, 0.999999],
                [0.999999, 0.000000],
                [0.676768, 0.000000],
                [0.000000, 0.090909],
                [0.999999, 0.999999],
                [0.000000, 0.6833],
                [0.000000, 0.000000],
                [0.999999, 0.3518],
                [0.000000, 0.999999],
                [0.999999, 0.000000],
                [0.000000, 0.4111],
            ]
        ),
        axis=0,
    )
    Y = np.append(
        Y,
        [
            0.0,
            0.0,
            5.56e-170,
            -1.43e-200,
            1.5176487295659e-192,
            1.7338671820466e-172,
            2.30881074212614e-248,
            -7.82323380643744e-151,
            0.0,
            0.0,
            -1.17138831964994e-124,
        ],
    )

    return X, Y


def function_2():
    X = np.load("./function_2/initial_inputs.npy")
    Y = np.load("./function_2/initial_outputs.npy")
    # Add some more points to the training data
    X = np.append(
        X,
        np.array(
            [
                [0.0, 0.999999],
                [0.999999, 0.999999],
                [0.000000, 0.696970],
                [0.323232, 0.999999],
                [0.7962, 0.0638],
                [0.7535, 0.999999],
                [0.6924, 0.5774],
                [0.682, 0.8801],
                [0.7079, 0.6451],
                [0.7017, 0.0752],
                [0.696, 0.000000],
            ]
        ),
        axis=0,
    )
    Y = np.append(
        Y,
        [
            0.02435325636,
            0.01112124947,
            0.06714351599,
            -0.01972122682,
            0.168081396980027,
            0.370651334460622,
            0.606365588790688,
            0.528032484364878,
            0.631636348480883,
            0.665731067398191,
            0.503479692159199,
        ],
    )
    return X, Y


def function_3():
    X = np.load("./function_3/initial_inputs.npy")
    Y = np.load("./function_3/initial_outputs.npy")
    # Add some more points to the training data
    X = np.append(
        X,
        np.array(
            [
                [0.999999, 0.999999, 0.999999],
                [0.999999, 0.0, 0.70707],
                [0.474747, 0.999999, 0.000000],
                [0.999999, 0.000000, 0.000000],
                [0.999999, 0.000000, 0.6371],
                [0.2465, 0.3125, 0.2409],
                [0.1693, 0.444, 0.2053],
                [0.9215, 0.2236, 0.7952],
                [0.3416, 0.3019, 0.3549],
                [0.9708, 0.3769, 0.2579],
                [0.9424, 0.9281, 0.6905],
            ]
        ),
        axis=0,
    )
    Y = np.append(
        Y,
        [
            -0.196020925114507,
            0.4949776013,
            -0.156923158387769,
            -0.189445466288635,
            -0.148349422894029,
            -0.117297137554259,
            -0.118336011626653,
            -0.0946650490996029,
            -0.0393096582333991,
            -0.1025822105944,
            -0.139822553679459,
        ],
    )
    return X, Y


def function_4():
    X = np.load("./function_4/initial_inputs.npy")
    Y = np.load("./function_4/initial_outputs.npy")
    # Add some more points to the training data
    X = np.append(
        X,
        np.array(
            [
                [0.416666, 0.375, 0.375, 0.416666],
                [0.416667, 0.375, 0.333333, 0.458333],
                [0.416667, 0.375, 0.333333, 0.416667],
                [0.458333, 0.458333, 0.0, 0.25],
                [0.3909, 0.4404, 0.4075, 0.4043],
                [0.4086, 0.3731, 0.4207, 0.4034],
                [0.4182, 0.3724, 0.3827, 0.3753],
                [0.4421, 0.3887, 0.4069, 0.4077],
                [0.3672, 0.3742, 0.3936, 0.3912],
                [0.3929, 0.371, 0.4223, 0.4232],
                [0.4003, 0.3621, 0.4162, 0.4092],
            ]
        ),
        axis=0,
    )
    Y = np.append(
        Y,
        [
            0.604199006086369,
            -0.384360639750433,
            0.330240283465102,
            -10.0412140228616,
            0.183437392548225,
            0.584816127817906,
            0.437824303331692,
            0.16155036820409,
            0.233869405760132,
            0.577365425282576,
            0.64308844267379,
        ],
    )
    return X, Y


def function_5():
    X = np.load("./function_5/initial_inputs.npy")
    Y = np.load("./function_5/initial_outputs.npy")
    # Add some more points to the training data
    X = np.append(
        X,
        np.array(
            [
                [0.666666, 0.999999, 0.999999, 0.999999],
                [0.999999, 0.999999, 0.999999, 0.999999],
                [0.999999, 0.999999, 0.999999, 0.999999],
                [0.999999, 0.999999, 0.999999, 0.999999],
                [0.9357, 0.4468, 0.3706, 0.8784],
                [0.9215, 0.8744, 0.513, 0.4307],
                [0.7779, 0.6849, 0.9202, 0.5694],
                [0.8185, 0.999, 0.8258, 0.2461],
                [0.9231, 0.9708, 0.2042, 0.3175],
                [0.6124, 0.4599, 0.6463, 0.2016],
                [0.1843, 0.9556, 0.7435, 0.6612],
            ]
        ),
        axis=0,
    )
    Y = np.append(
        Y,
        [
            5188.01687646763,
            8662.4050012483,
            8662.4050012483,
            8662.4050012483,
            638.323396105477,
            627.172317531108,
            635.123283742867,
            1465.24463708288,
            931.406266259295,
            7.55803829268376,
            527.682026379153,
        ],
    )
    return X, Y


def function_6():
    X = np.load("./function_6/initial_inputs.npy")
    Y = np.load("./function_6/initial_outputs.npy")
    # Add some more points to the training data
    X = np.append(
        X,
        np.array(
            [
                [0.0, 0.052631, 0.052631, 0.999999, 0.0],
                [0.315789, 0.0, 0.999999, 0.999999, 0.157895],
                [0.263158, 0.999999, 0.0, 0.999999, 0.0],
                [0.421054, 0.315789, 0.526316, 0.736842, 0.105263],
                [0.4635, 0.3707, 0.6235, 0.7525, 0.1564],
                [0.4549, 0.3513, 0.5761, 0.6678, 0.0299],
                [0.4299, 0.2846, 0.5324, 0.7952, 0.1921],
                [0.4039, 0.3954, 0.5006, 0.7289, 0.1875],
                [0.4765, 0.2831, 0.5944, 0.8598, 0.0798],
                [0.4537, 0.2224, 0.6314, 0.725, 0.1706],
                [0.4059, 0.3891, 0.6747, 0.8661, 0.1266],
            ]
        ),
        axis=0,
    )
    Y = np.append(
        Y,
        [
            -1.55976559584547,
            -1.01912518379867,
            -1.82858549587962,
            -0.154184369592684,
            -0.195207058457922,
            -0.348006059261988,
            -0.216896033801918,
            -0.377120756964277,
            -0.274445502791059,
            -0.332194033585449,
            -0.213995649058753,
        ],
    )
    return X, Y


def function_7():
    X = np.load("./function_7/initial_inputs.npy")
    Y = np.load("./function_7/initial_outputs.npy")
    # Add some more points to the training data
    X = np.append(
        X,
        np.array(
            [
                [0.0, 0.0, 0.111111, 0.0, 0.222222, 0.999999],
                [0.999999, 0.999999, 0.222222, 0.999999, 0.999999, 0.0],
                [0.0, 0.999999, 0.0, 0.0, 0.333333, 0.0],
                [0.0, 0.666667, 0.666667, 0.0, 0.333333, 0.999999],
                [0.0749, 0.423, 0.2999, 0.2338, 0.3802, 0.7253],
                [0.0827, 0.325, 0.3851, 0.2506, 0.3401, 0.7127],
                [0.0621, 0.2947, 0.4324, 0.2431, 0.315, 0.7177],
                [0.0291, 0.2743, 0.4426, 0.2389, 0.3334, 0.6884],
                [0.0392, 0.2741, 0.4438, 0.2857, 0.3315, 0.7308],
                [0.0782, 0.2842, 0.4786, 0.2096, 0.3406, 0.7228],
                [0.026, 0.2817, 0.4123, 0.2007, 0.3241, 0.7304],
            ]
        ),
        axis=0,
    )
    Y = np.append(
        Y,
        [
            0.341945960681604,
            0.00100349568909974,
            0.00815600414602021,
            0.455710235987283,
            1.94221294158671,
            2.63753976408422,
            2.74029949118323,
            2.72312468001456,
            2.70004488689188,
            2.72093043385009,
            2.48676483894626,
        ],
    )
    return X, Y


def function_8():
    X = np.load("./function_8/initial_inputs.npy")
    Y = np.load("./function_8/initial_outputs.npy")
    # Add some more points to the training data
    X = np.append(
        X,
        np.array(
            [
                [0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 0.25, 0.5],
                [0.25, 0.375, 0.0, 0.375, 0.375, 0.625, 0.5, 0.75],
                [0.125, 0.125, 0.25, 0.25, 0.5, 0.625, 0.5, 0.625],
                [0.375, 0.125, 0.125, 0.125, 0.375, 0.875, 0.375, 0.999999],
                [0.2055, 0.2709, 0.1903, 0.249, 0.6468, 0.5534, 0.2995, 0.7756],
                [0.2545, 0.1724, 0.1464, 0.2808, 0.7707, 0.5343, 0.3253, 0.553],
                [0.2432, 0.1428, 0.1461, 0.1517, 0.6199, 0.3739, 0.3408, 0.707],
                [0.2255, 0.28, 0.1577, 0.1071, 0.6858, 0.5937, 0.3339, 0.6251],
                [0.2817, 0.2514, 0.2269, 0.2379, 0.7456, 0.4617, 0.3306, 0.6349],
                [0.176, 0.1464, 0.1231, 0.2206, 0.6004, 0.5824, 0.2786, 0.659],
                [0.0895, 0.2609, 0.1139, 0.2616, 0.6798, 0.4247, 0.2919, 0.534],
            ]
        ),
        axis=0,
    )
    Y = np.append(
        Y,
        [
            9.8408,
            9.5148625,
            9.7042375,
            9.5392375799999,
            9.904942704,
            9.900615405,
            9.885241765,
            9.896235289,
            9.850611929,
            9.94389399,
            9.94402757,
        ],
    )
    return X, Y


def process(X, Y):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tensor_X = torch.from_numpy(X)
        tensor_Y = torch.from_numpy(Y)

        # Initialize and fit a Gaussian Process model
        gp_model = SingleTaskGP(tensor_X, tensor_Y.unsqueeze(-1))
        mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
        fit_gpytorch_mll(mll)

        # Define the acquisition function
        best_Y = (
            tensor_Y.max()
        )  # No need for .item() if further operations are in PyTorch
        EI = qExpectedImprovement(model=gp_model, best_f=best_Y)

        # Optimize the acquisition function to find new points
        bounds = torch.stack(
            [torch.zeros(tensor_X.size(-1)), torch.ones(tensor_X.size(-1))]
        )
        candidates, _ = optimize_acqf(
            acq_function=EI,
            bounds=bounds,
            q=1,  # Number of points to generate
            num_restarts=200,
            raw_samples=512
        )

        print("Candidates:", candidates)


## Completed until Module 19 - include result from 21 June

X, Y = function_1()
print("\nFunction 1 results:")
process(X, Y)

X, Y = function_2()
print("\n\nFunction 2 results:")
process(X, Y)

X, Y = function_3()
print("\n\nFunction 3 results:")
process(X, Y)

X, Y = function_4()
print("\n\nFunction 4 results:")
process(X, Y)

X, Y = function_5()
print("\n\nFunction 5 results:")
process(X, Y)

X, Y = function_6()
print("\n\nFunction 6 results:")
process(X, Y)

X, Y = function_7()
print("\n\nFunction 7 results:")
process(X, Y)

X, Y = function_8()
print("\n\nFunction 8 results:")
process(X, Y)
