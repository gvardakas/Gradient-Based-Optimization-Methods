#----------------------------------------------------------------
#		NAME			||	AM	||        e-mail
#	Georgios Vardakas	||	432	||  geoo1995@gmail.com
#   Dimitra Triantali   ||  431 ||  dimitra.triantali@gmail.com
#----------------------------------------------------------------
#	Course: Optimization
#   Project 1
# 	Written in Python 3.8.6

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv, cholesky, norm
from numpy.linalg.linalg import LinAlgError
import time
import sys

# Globals

# Our data for the first 30 days. Training set.
df_covid_data = pd.read_csv("./Data/covid_data_30_GR.dat", delim_whitespace=True, names = ["Day", "Cases"])
# Scaling step
df_covid_data["Scaled Day"] = df_covid_data["Day"].div(10)
df_covid_data["Scaled Cases"] = df_covid_data["Cases"].div(10000)

# Our data for days 13 - 17 of November based on https://covid19.who.int/region/euro/country/gr
df_covid_data_testset = pd.DataFrame(data = np.array([[31, 66637], [32, 69675], [33, 72510], [34, 74205], [35, 76403]]),
                          columns = ["Day", "Cases"])
# Scaling step
df_covid_data_testset["Scaled Day"] = df_covid_data_testset["Day"].div(10)
df_covid_data_testset["Scaled Cases"] = df_covid_data_testset["Cases"].div(10000)

# Defining the polynomial model, its gradiend and its hessian matrix
def model(a, x):
    return a[0] + a[1] * x + a[2] * np.power(x, 2) + a[3] * np.power(x, 3) + a[4] * np.power(x, 4)

def model_gradient(a, x):
    gradient = np.array([[1], [x], [np.power(x, 2)], [np.power(x, 3)], [np.power(x, 4)]])
    return gradient

def model_hessian(a, x):
    hessian = np.zeros(shape=(5, 5))
    return hessian

# Defining the error function, its first and its second derivative
def error_function(model_parameters):
    total_error = 0
    for index, row in df_covid_data.iterrows():
        scaled_day, real_scaled_cases = row[["Scaled Day", "Scaled Cases"]]
        model_error_i = model(model_parameters, scaled_day) - real_scaled_cases
        total_error += np.power(model_error_i, 2)
    return total_error

def gradient_of_error_function(model_parameters):
    gradient = np.zeros(shape=model_parameters.shape)
    for index, row in df_covid_data.iterrows():
        scaled_day, real_scaled_cases = row[["Scaled Day", "Scaled Cases"]]
        gradient_i = 2 * (model(model_parameters, scaled_day) - real_scaled_cases) * model_gradient(model_parameters, scaled_day)
        gradient = np.add(gradient, gradient_i)
    return gradient

def hessian_of_error_function(model_parameters):
    hessian = np.zeros(shape=(model_parameters.shape[0], model_parameters.shape[0]))
    for index, row in df_covid_data.iterrows():
        scaled_day, real_scaled_cases = row[["Scaled Day", "Scaled Cases"]]
        hessian_i = 2 * (model_gradient(model_parameters, scaled_day) @ np.transpose(model_gradient(model_parameters, scaled_day))
                        + model(model_parameters, scaled_day) * model_hessian(model_parameters, scaled_day)
                        - real_scaled_cases * model_hessian(model_parameters, scaled_day))
        hessian = np.add(hessian, hessian_i)
    return hessian

# Cholesky method
def cholesky_method(hessian):
    # If he is then the inverse of hessian is positive defiend too
    # If not then cholesky will raise error
    try:
        cholesky(hessian)
        return True
    except LinAlgError as err:
        print(err)
        return False

## Hessian matrix modification
def hessian_modification(hessian_xk, beta):
    diagonal_elemets = np.diag(hessian_xk)
    min_diag_element = np.min(diagonal_elemets)
    I = np.identity(hessian_xk.shape[0])
    max_iteration = 100

    t = 0
    if (min_diag_element <= 0):
        t = -min_diag_element + beta

    for i in range(max_iteration):
        try:
            L = cholesky(np.add(hessian_xk, t * I))
            return L
        except LinAlgError as err:
            t = max(t * 2, beta)
    return I

# Routines needed for line search methods
# Strong Wolf Conditions
def strong_wolf_conditions(pk, xk, a_max = 2, c2 = 1e-1):
    c1 = 10e-4
    ai_low = 0
    ai = interpolate(ai_low, a_max)
    iteration = 0
    line_a = lambda xk, a, pk: np.add(xk, a * pk)

    while True:
        # f(xk + a * pk)
        f_ai = error_function(line_a(xk, ai, pk))[0]

        # f(xk + 0 * pk)
        f_0 = error_function(xk)[0]

        # f'(xk + 0 * pk)
        f_gradient_0 = (np.transpose(gradient_of_error_function(xk)) @ pk)[0][0]

        # f(xk + ai_low * pk)
        f_ai_low = error_function(line_a(xk, ai_low, pk))[0]

        # Armijo condition false
        armijo_condition = f_ai > f_0 + c1 * ai * f_gradient_0

        if ((armijo_condition) or (f_ai >= f_ai_low and iteration > 0)):
            #print("A")
            return zoom(c1, c2, line_a, f_0, f_gradient_0, pk, xk, ai_low, ai)

        # f'(xk + ai * pk)
        f_gradient_ai = (np.transpose(gradient_of_error_function(line_a(xk, ai, pk))) @ pk)[0][0]

        # Curvature condition
        curvature_condition = abs(f_gradient_ai) <= -c2 * f_gradient_0

        # If both conditions true return ai
        if(curvature_condition):
            return ai

        if(f_gradient_ai >= 0):
            return zoom(c1, c2, line_a, f_0, f_gradient_0, pk, xk, ai, ai_low)

        ai_low = ai
        ai = interpolate(ai_low, a_max)
        iteration += 1

def zoom(c1, c2, line, f_0, f_gradient_0, pk, xk, a_low, a_high):
    iteration = 0
    max_iteration = 100
    while True:
        aj = interpolate(a_low, a_high)

        # f(xk + aj * pk)
        f_aj = error_function(line(xk, aj, pk))[0]

        # Armijo condition false
        armijo_condition = f_aj > f_0 + c1 * aj * f_gradient_0

        # f(xk + a_low * pk)
        f_a_low = error_function(line(xk, a_low, pk))[0]

        # If armijo condition false make a_high smaller
        if ((armijo_condition) or (f_aj >= f_a_low)):
            a_high = aj
        else:
            # f'(xk + aj * pk)
            f_gradient_aj = (np.transpose(gradient_of_error_function(line(xk, aj, pk))) @ pk)[0][0]

            # Curvature condition
            curvature_condition = abs(f_gradient_aj) <= -c2 * f_gradient_0

            # If both conditions true return aj
            if(curvature_condition):
                return aj

            # If gradient is positive then make a_high smaller
            if(f_gradient_aj * (a_high - a_low) >= 0):
                a_high = a_low

            a_low = aj

        iteration += 1
        if(iteration > max_iteration):
            return interpolate(a_low, a_high)

def interpolate(a_low, a_high):
    # Using Bisection
    return (a_low + a_high) / 2

# Line search methods
def steepest_descent_wolf_conditions(init_parameters):
    iteration = 0
    max_iterations = 1000
    xk = init_parameters
    error_list = list()
    error_list.append(error_function(xk))

    while (iteration < max_iterations and norm(gradient_of_error_function(xk)) > 1e-6):
        # Calculating the gradient
        gradient_xk = gradient_of_error_function(xk)

        # Steepest descent as direction
        pk = -gradient_xk

        # Chosing step size, choosing c2=1e-1 for steepest_descent
        a = strong_wolf_conditions(pk, xk, c2 = 1e-1)

        # Updating the new xk_1
        xk_1 = np.add(xk, a * pk)

        # Calculating error just for ploting
        error = error_function(xk_1)
        error_list.append(error)

        print("Iteration: %d, Step: %.1E, Error: %.4f" % (iteration, a, error))

        # Update iteration and xk
        iteration += 1
        xk = xk_1

    return xk, error_list

def newton_direction_wolf_conditions(init_parameters):
    iteration = 0
    max_iterations = 1000
    xk = init_parameters
    error_list = list()
    error_list.append(error_function(xk))

    while (iteration < max_iterations and norm(gradient_of_error_function(xk)) > 1e-6):
        # Calculating the gradient and the hessian
        gradient_xk = gradient_of_error_function(xk)
        hessian_xk = hessian_of_error_function(xk)

        # Checking if hessian is positive defined
        pos_defined = cholesky_method(hessian_xk)

        # If hessian is not positive defined then modify it
        if (not pos_defined):
            hessian_xk = hessian_modification(hessian_xk, beta=1)

        # Newton direction
        pk = -inv(hessian_xk) @ gradient_xk

        # Chosing step size, choosing c2=9e-1 for newton diration
        a = strong_wolf_conditions(pk, xk, c2=9e-1)

        # Updating the new xk_1
        xk_1 = np.add(xk, a * pk)

        # Calculating error just for ploting
        error = error_function(xk_1)
        error_list.append(error)

        print("Iteration: %d, Step: %.1E, Error: %.4f" % (iteration, a, error))

        # Update iteration and xk
        iteration += 1
        xk = xk_1

    return xk, error_list

def BFGS_wolf_conditions(init_parameters, hessian_approx):
    iteration = 0
    max_iterations = 1000
    I = np.identity(init_parameters.shape[0])
    hessian_xk = hessian_approx
    xk = init_parameters
    error_list = list()
    error_list.append(error_function(xk)[0])

    while (iteration < max_iterations and norm(gradient_of_error_function(xk)) > 1e-6):
        # Calculating the gradient and the hessian
        gradient_xk = gradient_of_error_function(xk)

        # Checking if hessian is positive defined only in the first iteration
        if(iteration == 0):
            pos_defined = cholesky_method(hessian_xk)
            if (not pos_defined):
                hessian_xk = hessian_modification(hessian_xk, beta=1)

        # Newton direction
        pk = -hessian_xk @ gradient_xk

        # Chosing step size
        a = strong_wolf_conditions(pk, xk, c2 = 1e-1)

        # Updating the new xk_1
        xk_1 = np.add(xk, a * pk)

        # Calculating sk
        sk = np.subtract(xk_1, xk)

        # Calculating the next gradient
        gradient_xk_1 = gradient_of_error_function(xk_1)

        # Calculating yk
        yk = np.subtract(gradient_xk_1, gradient_xk)

        # BFGS method for updating the Hessian
        # Calculating rk

        # To solve problem with numbers close to zero
        if (np.transpose(yk) @ sk <= 0):
            break

        rk = 1 / (np.transpose(yk) @ sk)

        # Update Hessian
        matrix_1 = (I - rk * sk @ np.transpose(yk))
        matrix_2 = (I - rk * yk @ np.transpose(sk))
        hessian_xk = matrix_1 @ hessian_xk @ matrix_2 + rk * sk @ np.transpose(sk)

        # Calculating error just for ploting
        error = error_function(xk_1)
        error_list.append(error[0])

        print("Iteration: %d, Step: %.1E, Error: %.4f" % (iteration, a, error))

        # Update iteration, xk and gradient_prev
        iteration += 1
        xk = xk_1

    return xk, error_list

# Routines needed for trust region methods
def get_direction(xk, Bk, deltak):
    # Calculating gradient at k just one time
    gradient_k = gradient_of_error_function(xk)

    pB = -inv(Bk) @ gradient_k

    # If inside trust region
    if (norm(pB, 2) <= deltak):
        return pB, "Newton"
    # If outside trust region
    else:
        # Cauchy point
        pU = -((np.transpose(gradient_k) @ gradient_k)
            / (np.transpose(gradient_k) @ Bk @ gradient_k)) * gradient_k
        # If outside trust region, cut it at trust region border
        if (norm(pU, 2) >= deltak):
            return -(deltak / norm(gradient_k, 2) * gradient_k), "Cauchy"
        # If inside trust region, do dogleg
        else:
            t = solve(pB, pU, deltak)
            return pU + (t - 1) * np.subtract(pB, pU), "Dogleg"

# Bisection method
def solve(pB, pU, deltak):
    iteration = 0
    t_lower = 1
    t_upper = 2
    error = 1e-9
    equation = lambda ti: np.power(norm(pU + (ti - 1) * (pB - pU), 2), 2) - np.power(deltak, 2)

    while True:
        iteration += 1
        ti = (1 / 2) * (t_upper + t_lower)

        if (abs(equation(ti)) <= error):
            return ti
        else:
            if (equation(t_lower) * equation(ti) < 0):
                t_upper = ti
            else:
                t_lower = ti

            if(t_upper - t_lower < error):
                return ti

# Thrust region method
def newton_method_safe_region(init_parameters):
    iteration = 0
    max_iterations = 1000
    delta_low, delta_max = 0, 1
    delta_i = (delta_low + delta_max) / 2
    htta = 1 / 4
    xk = init_parameters
    step = lambda xk, pk: np.add(xk, pk)
    mk = lambda xk, pk, Bk: error_function(xk) + (np.transpose(gradient_of_error_function(xk)) @ pk)[0][0] + ((1 / 2) * np.transpose(pk) @ Bk @ pk)[0][0]
    error_list = list()
    error_list.append(error_function(xk))
    point_counter = {"Newton": 0, "Cauchy" : 0, "Dogleg" : 0}

    while (iteration < max_iterations and norm(gradient_of_error_function(xk)) > 1e-6):
        # Calculating hessian
        hessian_xk = hessian_of_error_function(xk)

        # Checking if hessian is positive defined
        pos_defined = cholesky_method(hessian_xk)
        if (not pos_defined):
            hessian_xk = hessian_modification(hessian_xk, 1)

        # Chosing the descent direction
        pk, point = get_direction(xk, hessian_xk, delta_i)
        point_counter[point] += 1

        # rk quantifies the quality of the solution
        real_reduction = error_function(xk) - error_function(step(xk, pk))
        approximate_reduction = mk(xk, np.zeros(pk.shape), hessian_xk) - mk(xk, pk, hessian_xk)
        rk = real_reduction / approximate_reduction

        # Fixing the trust region for next iteration
        # Case rk close to 0
        if (rk < 1 / 4):
            delta_i = (1 / 4) * delta_i
        # Case rk close to 1
        elif ((rk > 3 / 4) and (norm(pk, 2) > delta_i)):
            delta_i = min(2 * delta_i, delta_max)
        # Case where 0 << rk < 1
        else:
            delta_i = delta_i

        # Checking if we accept the step or not
        # Step pk accepted
        if (rk > htta):
            xk_1 = np.add(xk, pk)
            # Calculating error just for ploting
            error = error_function(xk_1)
            error_list.append(error)
            print("Iteration: %d, Error: %.4f" % (iteration, error))
        # Step pk rejected
        else:
            xk_1 = xk

        # Update iteration and xk
        iteration += 1
        xk = xk_1

    print("\nNewton point: %s, Cauchy point: %s, Dogleg point: %s." %
        (point_counter["Newton"], point_counter["Cauchy"], point_counter["Dogleg"]))
    return xk, error_list

def optimizer(method_choice, init_parameters):
    time_1 = time.time()

    if (method_choice == "1"):
        print("Method: Newton with strong wolfe conditions.\n")
        parameters, errors = newton_direction_wolf_conditions(init_parameters)
    elif(method_choice == "2"):
        print("Method: Newton with trust region.\n")
        parameters, errors = newton_method_safe_region(init_parameters)
    elif(method_choice == "3"):
        print("Method: BFGS with strong wolfe conditions.\n")
        hessian_approx = np.identity(init_parameters.shape[0])
        parameters, errors = BFGS_wolf_conditions(init_parameters, hessian_approx)
    elif(method_choice == "4"):
        print("Method: Steepest descent with strong wolfe conditions.\n")
        parameters, errors = steepest_descent_wolf_conditions(init_parameters)

    time_2 = time.time()
    time_difference = time_2 - time_1
    print("\nTime passed in secods: %.2f.\n" % (time_difference))
    return parameters, errors

# Model evaluation
def mean_squared_error(y_true, y_pred):
    squared_error = 0
    for index, _ in enumerate(y_true):
        squared_error += np.power((y_true[index] - y_pred[index]), 2)
    mse = squared_error / y_true.shape[0]
    return mse

def plot_model_results(parameters):
    answers_trainingset = []
    predictions = []
    #for mse
    predictions_no_scaling = []
    scale_day = 10
    scale_cases = 10000
    # divition with 10 for scaling
    data_range = np.arange(0, 31, 1) / scale_day
    predictions_range = np.arange(30, 36, 1) / scale_day

    for i in data_range:
        # multiplication with 10000 for scaling
        answers_trainingset.append(model(parameters, i) * scale_cases)

    for i in predictions_range:
        # multiplication with 10000 for scaling
        predictions.append(int((model(parameters, i) * scale_cases)[0]))
        predictions_no_scaling.append(model(parameters, i))

    print("Parameters after optimization: %s.\n" % (np.array2string(np.transpose(parameters))))

    mse = mean_squared_error(np.asarray(predictions_no_scaling[1:]), df_covid_data_testset["Scaled Cases"].to_numpy())
    print("Mean squared error (at testing set, without rescaling): %.4f.\n" % (mse))
    data = {"Day Index":df_covid_data_testset["Day"].to_list(),
     "Real Cases": df_covid_data_testset["Cases"].to_list(),
     "Model Prediction": predictions[1:]}

    df_real_predictions = pd.DataFrame(data=data)
    df_real_predictions["Error"] = (df_real_predictions["Real Cases"] - df_real_predictions["Model Prediction"]).abs()
    print(df_real_predictions)
    print("")

    title = "Trained model"
    fontsize = 15
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 8)
    ax.scatter(df_covid_data["Day"], df_covid_data["Cases"], c="r")
    ax.scatter(df_covid_data_testset["Day"], df_covid_data_testset["Cases"], c="b")
    ax.plot(data_range * scale_day, answers_trainingset)
    ax.plot(predictions_range * scale_day, predictions)
    ax.set_xlabel("Days since 14/10/2020", fontsize=fontsize)
    ax.set_ylabel("Cumulative covid-19 cases", fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    ax.grid(True)
    ax.legend(["Fitted model", "Model's Prediction", "Training Set", "Testing Set"], fontsize=fontsize)
    fig.savefig(title, facecolor='w')
    plt.show()


def main():
    if(len(sys.argv) != 5):
        print("Error: wrong input.")
        print("Usage: python3 optimization.py m method_number x initial_parameter_number")
        print("method_number choices: 1, 2, 3, 4.")
        print("initial_parameter_number choices: 0, 1, 2, 3, 4")
        print("Example: python3 optimization.py m 1 x 0")

    method_choice = sys.argv[2]
    parameters_choice = sys.argv[4]

    # possible choices for initial parameters
    x0 = np.array([[5.025], [-4.898], [0.119], [3.981], [7.818]])
    x1 = np.array([[9.185], [0.944], [-7.227], [-7.014], [-4.849]])
    x2 = np.array([[6.814], [-4.914], [6.285], [-5.129], [8.585]])
    x3 = np.array([[-2.966], [6.616], [1.705], [0.994], [8.343]])
    x4 = np.array([[-7.401], [1.376], [-0.612], [-9.762], [-3.257]])
    parameters_dict = {"0" : x0, "1" : x1, "2" : x2, "3" : x3, "4" : x4}
    init_parameters = parameters_dict[parameters_choice]

    print("Initial model parameters: x%s = %s.\n" %
        (parameters_choice, np.array2string(np.transpose(init_parameters))))

    parameters, errors = optimizer(method_choice, init_parameters)

    plot_model_results(parameters)

if __name__ == "__main__":
	main()
