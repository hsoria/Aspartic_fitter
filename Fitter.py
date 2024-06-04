import streamlit as st
import pandas as pd


from scipy.integrate import odeint, solve_ivp, lsoda
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from lmfit import Parameters, minimize, Model, report_fit, conf_interval
from sklearn.metrics import mean_squared_error
import numdifftools
from PIL import Image
from sklearn.metrics import r2_score
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import scipy.optimize
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as tkr






def kinetic_plotting(z, t, params):
    F, Ac, An, W = z  
    k0, k1, a, k4 = params
    
    dAcdt = -k1*F*Ac + k4*An + ((k1*a*Ac*F)/(a+1))
    dFdt = -k1*F*Ac - k0*F
    dWdt = +k1*F*Ac + k0*F
    dAndt = +((k1*Ac*F)/(a+1)) - k4*An
    
    return [dFdt, dAcdt, dAndt, dWdt]


def ode_model(z, t, k0, k1, a, k4):
    
    """
    takes a vector of the initial concentrations that are previously defined. 
    YOu have to provide also initial guesses for the kinetic constants. 
    You must define as much constants as your system requires. Note that the reactions are expressed as differential
    equations
    
    F: Fuel
    Ac: Precuros
    An: Anhydride
    W: Waste
    
    Time is considered to be in **minutes**. Concentrations are in **mM**
    
    """
    F, Ac, An, W = z  
    
    dAcdt = -k1*F*Ac + k4*An + ((k1*a*Ac*F)/(a+1))
    dFdt = -k1*F*Ac - k0*F
    dWdt = k1*F*Ac + k0*F
    dAndt = ((k1*Ac*F)/(a+1)) - k4*An
    
    
    return [dFdt,dAcdt,dAndt,dWdt]

def ode_solver(t, initial_conditions, params):
    
    """
    Solves the ODE system given initial conditions for both initial concentrations and initial guesses for k

    
    """
        
    F, Ac, An, W = initial_conditions
    k0, k1, a, k4 = params['k0'].value, params['k1'].value, params['a'].value, params['k4'].value
    res = odeint(ode_model, [F, Ac, An, W], t, args=(k0, k1, a, k4))
    return res

def error(params, initial_conditions, tspan, data):
    sol = ode_solver(tspan, initial_conditions, params)
    sol_subset = sol[:, [0, 1, 2,]]
    data_subset = data[:, [0, 1, 2,]]
    return (sol_subset-data_subset)





def load_data_frame(excel_name, sheet_name):
    """
    Loads the excel file with data sortened in different sheets. Each sheet corresponds to different parameter change. 


    """
    a = pd.read_excel(excel_name, sheet_name= f"{sheet_name}", skiprows=(range(0,2)))
    a.columns = ["time", "F", "Ac", "An", "W", "Condition"] 

    a = a.sort_values(by = "time")

    return a



def load_initial_conditions(df, k0):
    
    
    
    """
    
    Provided a dataframe we extract the initial concentrations, the time window and provide initial guesses for the kinetic modelling. 
    Note that one can modifify the boundaries of the kinetic constants.
    Provide a df with the proper formatting
    
    """
    tspan = np.linspace(df["time"][0],float(df["time"].iloc[-1])+20,1000)
    F = df["F"][0] 
    Ac = df["Ac"][0]
    An = df["An"][0] 
    W = df["W"][0] 

    initial_conditions = [F, Ac, An, W]
    
    #initial guesses.
    
    k1 = 0.1      
    a = .1 #k2/k3
    k4 = .1  

    params = Parameters()
    params.add('k0', value=k0, vary = False)
    params.add('k1', value=k1, min=1e-6, max= 10)
    params.add('a', value=a, min=1e-6, max= 10)
    params.add('k4', value=k4, min=1e-6, max= 10)
    
    return initial_conditions, params, tspan

def sort_condition(df):

    list_df = []
    conditions = df["Condition"].unique()
    for condition in conditions:
        g = df[df["Condition"] == condition].sort_values(by = ["time", "Condition"]).reset_index(drop = True)
        list_df.append(g)

    return list_df, conditions
    

def get_fitted_curve(initial_conditions, tspan, params):
    """
    You provide the initial conditions, and the fitted values for the kinetic constants to simulate data.
    Is the "line" in the fitting curves.
    
    initial_conditions 
    tspan: time window simulation
    params: fitted parameters
    
    returns fitted data
    """
    
    y = pd.DataFrame(odeint(kinetic_plotting, initial_conditions, tspan, args=(params,)), columns = ['F', 'Ac', 'An','W'])
    y['min'] = tspan
    return y 




def fit_data(data_to_fit, fit_method = "least_squares", k0 = 1.3600e-04):

    initial_conditions, params, tspan = load_initial_conditions(data_to_fit, k0)

    data_to_fit = data.sort_values(by = "time")
    data = data_to_fit[['F', 'Ac', 'An', 'W']].values
    t = data_to_fit['time']
    result = minimize(error, 
                        params, 
                        args=(initial_conditions, t, data), 
                        method=fit_method, nan_policy='omit')
        
    
    # Extract parameter values
    k_values = pd.DataFrame({"k_values": result.params.valuesdict().values()})
    
    simulated_data = get_fitted_curve(initial_conditions,
                                       tspan = tspan, 
                                       params = k_values.values.flatten())



    return k_values, data_to_fit, simulated_data, tspan



def plot_fitted(df, y):

    colors = [  "#56B4E9", "#009E73", "#CC79A7", "#999999", "#E69F00","#DB2B39", "#0076A1", "#0072B2", "#1A5042","#0C1713"]
    palette = sns.color_palette(colors)

    sns.set_theme(context='notebook', style='ticks', 
                  font_scale=1.3, 
                  rc={"lines.linewidth": 1.6, 'axes.linewidth': 1.6, 
                      "xtick.major.width": 1.6, "ytick.major.width": 1.6}, 
                  palette=palette)


    fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize = (8, 3), 
                                    sharey = False, sharex = True,
                                      dpi = 300)
    sns.scatterplot(data = df, x = 'time', y = 'F', ax = ax1, color ="C0")
    sns.lineplot(data = y, x = 'min', y = 'F', ax = ax1, alpha = 0.5, color = "C0")

    sns.scatterplot(data = df, x = 'time', y = 'Ac', ax = ax2, color = "C1")
    sns.lineplot(data = y, x = 'min', y = 'Ac', ax = ax2, alpha = 0.5, color = "C1")

    sns.scatterplot(data = df, x = 'time', y = 'An', ax = ax3, color = "C2")
    sns.lineplot(data = y, x = 'min', y = 'An', ax = ax3, alpha = 0.5, color = "C2")




    ax1.set(xlabel = 'Time [min]', ylabel = 'EDC [mM]')
    ax2.set(xlabel = 'Time [min]', ylabel = 'Acid [mM]')
    ax3.set(xlabel = 'Time [min]', ylabel = 'Anhydride [mM]')




    plt.tight_layout(pad = 1.08, w_pad=2)

    # Display the plot using Streamlit
    st.pyplot(fig)
    
    return fig, (ax1, ax2, ax3)







def fit_bootstrapping(data_to_fit, fit_method = "least_squares", k0 = 1.3600e-04, n_bt = 100):
    
    """
    We sample our data n times doing fitting of each sampled data. We store the solutions for the k values. 
    In the end we obtain a distribution of kinetic constants. 
    One can also calcualate other parameters, for instance the half-life of the anhydride.
    
    
    n_iter: number of bootstrapping steps. 
    df: parent dataframe
    method: Minimization method
    error: Loss function
    initial_conditions, 
    params, 
    tspan
    
    """
    initial_conditions, params, tspan = load_initial_conditions(data_to_fit, k0)


    data = data_to_fit[['F', 'Ac', 'An', 'W']].values
    t = data_to_fit['time']


    
    k1_bt = []
    a_bt = []
    k4_bt = []
    half_life = []

    
        
    for i in range(0, n_bt):
        
        """
        This is the most crucial sentence of the code. 
        It minimize out error function according to our initial values, guesses and time window. 
        There are several algorithms to minimize the function.
        
        Here's a list of those available. Note that some of them may require long computation period. 
        https://lmfit.github.io/lmfit-py/fitting.html
        
        """
        
        
        bt = data_to_fit.sample(n = len(data_to_fit), replace=True).reset_index(drop = True)
        bt = bt.sort_values(by = "time")
        data = bt[[ 'F', 'Ac', 'An','W']].values
        t = bt['time']
        result = minimize(error, 
                          params, 
                          args=(initial_conditions, t, data), 
                          method=fit_method, nan_policy='omit')
        
        k1_bt.append(result.params['k1'].value)
        a_bt.append(result.params['a'].value)
        k4_bt.append(result.params['k4'].value)
        half_life.append(np.log(2)/(result.params['k4'].value))

        k_values_bt = pd.DataFrame({"k1":k1_bt,
                           "a":a_bt,
                           'k4':k4_bt,
                            "half-life":half_life
                           })
        k_values_bt["k0"] = k0
        params.update(result.params)

       # Extract parameter values
        k0 = k0 
        k1 = float(k_values_bt.k1.median())
        a = float(k_values_bt.a.median())
        k4 = float(k_values_bt.k4.median())


        k_values = pd.DataFrame({"k_values_median": [k0, k1, a, k4]})
        
        simulated_data = get_fitted_curve(initial_conditions,
                                        tspan = tspan, 
                                        params = k_values.values.flatten()) 


    return k_values, data_to_fit, simulated_data, tspan, k_values_bt

def get_k_fitted(statistic, res):
    
    
    """
    
    Provides the mean/median value of the distribution of the kinetic constants
    statistics: mean, median
    res: result of the fitting
    
    """
    k0 = 1.3600e-04 
    if statistic == "mean":
        k1 = float(res.k1.mean())
        a = float(res.a.mean())
        k4 = float(res.k4.mean())
    elif statistic == "median":
        k1 = float(res.k1.median())
        a = float(res.a.median())
        k4 = float(res.k4.median())
    else:
        print("I do not understand you")

    params = k0,k1,a,k4
    
    return params

def get_rmse(data_to_fit, params_fitted):
    
    """
    
    Calculates the goodness of the fit through the R2. Which is a well-known parameter to decide whether a fit
    was done properly. Ranges between 0 and 1. Close to 1 equals to good fitting. 
    
    initial_conditions 
    df_real: original data
    params: fitted parameters
    
    returns fitted data
    
    """

    y_real = data_to_fit[['F', 'Ac', 'An']]
    t = data_to_fit['time']
    y_predict = pd.DataFrame(odeint(kinetic_plotting, 
                                    y_real.iloc[0].to_list() + [0], 
                                    t, 
                                    args = (params_fitted,)), 
                                     columns = ['F', 'Ac', 'An', "W"])
    
    rmse_F = np.sqrt(np.mean((y_predict["F"] - y_real["F"])**2, axis=0))
    rmse_Ac = np.sqrt(np.mean((y_predict["Ac"] - y_real["Ac"])**2, axis=0))
    rmse_An = np.sqrt(np.mean((y_predict["An"] - y_real["An"])**2, axis=0))
    
    rmse = [rmse_F, rmse_Ac, rmse_An]

    return rmse


def get_r2(data_to_fit, params_fitted):
    
    """
    
    Calculates the goodness of the fit through the R2. Which is a well-known parameter to decide whether a fit
    was done properly. Ranges between 0 and 1. Close to 1 equals to good fitting. 
    
    initial_conditions 
    df_real: original data
    params: fitted parameters
    
    returns fitted data
    
    """
    

    y_real = data_to_fit[['F', 'Ac', 'An']]
    t = data_to_fit['time']
    y_predict = pd.DataFrame(odeint(kinetic_plotting, 
                                    y_real.iloc[0].to_list() + [0], 
                                    t, 
                                    args = (params_fitted,)), 
                                    columns = ['F', 'Ac', 'An', "W"])
    
    r2_F = round(r2_score(y_real["F"], y_predict["F"]),4)
    r2_Ac = round(r2_score(y_real["Ac"], y_predict["Ac"]),4)
    r2_An = round(r2_score(y_real["An"], y_predict["An"]),4)
    
    r2 = [r2_F, r2_Ac, r2_An]

    return r2

def plot_fitted_error_2(df, y,rmse, tsimulation):
    
    
    """
    Creates a 4 column plots. In each column there is a different reagent. It plots both the original data and
    the fitted. Error is the 95 Confidence interval of the median. It is showed as filled area.
    
    df: data
    y: fitted
    y_max: upper limit
    y_min: low limit
    
    
    """
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize = (8, 3), 
                                    sharey = False, sharex = True)
    sns.scatterplot(data = df, x = 'time', y = 'F', ax = ax1, color = "C0")
    sns.lineplot(data = y, x = 'min', y = 'F', ax = ax1, alpha = 0.5, color = "C0", estimator = None)
    ax1.fill_between(tsimulation, y["F"]+(1.96*rmse[0]), y["F"]-(1.96*rmse[0]), alpha = 0.2, color = "C0", lw = 0)

    sns.scatterplot(data = df, x = 'time', y = 'Ac', ax = ax2, color = "C1")
    sns.lineplot(data = y, x = 'min', y = 'Ac', ax = ax2, alpha = 0.5, color = "C1", estimator = None)
    ax2.fill_between(tsimulation, y["Ac"]+(1.96*rmse[1]), y["Ac"]-(1.96*rmse[1]), alpha = 0.2, color = "C1", lw = 0)


    sns.scatterplot(data = df, x = 'time', y = 'An', ax = ax3, color = "C2")
    sns.lineplot(data = y, x = 'min', y = 'An', ax = ax3, alpha = 0.5, color = "C2", estimator = None)
    ax3.fill_between(tsimulation, y["An"]+(1.96*rmse[2]), y["An"]-(1.96*rmse[2]), alpha = 0.2, color = "C2", lw = 0)



    ax1.set(xlabel = 'Time [min]', ylabel = 'EDC [mM]', xticks = np.linspace(0, tsimulation[-1], 3))
    ax2.set(xlabel = 'Time [min]', ylabel = 'Acid [mM]')
    ax3.set(xlabel = 'Time [min]', ylabel = 'Anhydride [mM]')



    plt.tight_layout(pad = 1.08, w_pad=2)


    # Display the plot using Streamlit
    st.pyplot(fig)
    
    return fig, (ax1, ax2, ax3)

def streamlit_main():
    st.markdown("# Kinetic fitter")
    st.markdown("## Upload excel file with the experimental data and then the fitting is done")
    st.markdown(r"### **You can find a template for the excel file [here](https://github.com/hsoria)**")
    st.markdown(r"- You can fit multiple conditions as long as you have them sorted as in the template format")
    st.markdown(r"- If you only have one conditions, the fitting will only work that condition.")
    st.markdown(r"- If you want to fit data with different conditions we recommend to do individual fittings")

    
    st.markdown("## Kinetic rates")
    st.markdown("1. k0: F➝W $min^{-1}$")
    st.markdown("2. k1: F + Ac ➝ O $mM^{-1} min^{-1}$")
    st.markdown("3. k2: O ➝ An + W")
    st.markdown("4. k3: O ➝ Ac + W")
    st.markdown("5. k4: An ➝ Ac $min^{-1}$")
    st.markdown(r"We assume [O] ~ 0  so a = $$\frac{k3}{k2}$$")

    


    # File uploader for Excel file
    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

    # Text input for k0 value
    k0_input = st.number_input("Enter value for k0", format="%.10f")

    if uploaded_file is not None and k0_input:
        # Load data from uploaded Excel file
        df = load_data_frame(uploaded_file, sheet_name="Sheet1")
        dfs = sort_condition(df)

        df_list = dfs[0]
        conditions = dfs[1]

        # Convert k0_input to float
        k0_value = float(k0_input)


        # Perform fitting for each condition
        results = []
        k_values_fit = []
        r2_results = []
        for condition_df in df_list:
            k_values, original_data, simulated_data, tspan = fit_data(condition_df, k0=k0_value)
            results.append((k_values, original_data, simulated_data, tspan))
            c = original_data["Condition"].iloc[0]
            k = k_values
            k.columns = [f"{c}"]
            k_values_fit.append(k)
            # Plotting
        
            plot_fitted(original_data, simulated_data)
            r2 = pd.DataFrame(get_r2(data_to_fit=df, 
                                     params_fitted=k_values.values.flatten()), 
                              columns=[f"{c}"], 
                              index=["EDC", "Ac", "An"])
            
            r2_results.append(r2)


    # Table 1: Parameters

        
        parameters_table = pd.concat(objs = k_values_fit, axis = 1)
        
    

        parameters_table.index = ["k0", "k1", "a", "k4"]
        centered_html = f"<div style='width: 100%; text-align: center;'>{parameters_table.to_html(index=True)}</div>"
        
        # Table 2: Fitting results
        fitting_results = pd.concat(objs = r2_results, axis = 1)

        centered_html2 = f"<div style='width: 100%; text-align: center;'>{fitting_results.to_html(index=True)}</div>"

        # Display tables side by side
        col1, col2 = st.columns(2)

        # Table 1: Parameters
        
        with col1:
            st.markdown("## Parameters fitted")
            st.markdown(centered_html, unsafe_allow_html=True)

        # Table 2: Fitting results
        
        with col2:
            st.markdown(r"## R$^{2}$")

            st.markdown(centered_html2, unsafe_allow_html=True)
        
if __name__ == "__main__":
    streamlit_main()
