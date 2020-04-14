# -*- coding: utf-8 -*-

try:
    import pandas as pd
    import numpy as np
    import pmdarima as pm
    #%matplotlib inline
    import matplotlib.pyplot as plt
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.arima_model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import acf
    from dateutil.parser import parse
except ImportError as e:
    print("[FAILED] {}".format(e))

class operationsArima():

    @staticmethod
    def ForecastingWithArima():

        try:

            df = pd.read_csv('data.csv', names=['value'], header=0)

            model = pm.auto_arima(df.value, start_p=1, start_q=1,
                                test='adf',       # use adftest to find optimal 'd'
                                max_p=3, max_q=3, # maximum p and q
                                m=1,              # frequency of series
                                d=None,           # let model determine 'd'
                                seasonal=False,   # No Seasonality
                                start_P=0, 
                                D=0, 
                                trace=True,
                                error_action='ignore',  
                                suppress_warnings=True, 
                                stepwise=True)

            print(model.summary())

            model.plot_diagnostics(figsize=(7,5))
            plt.show()

            # Forecast
            n_periods = 24
            fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
            index_of_fc = np.arange(len(df.value), len(df.value)+n_periods)

            # make series for plotting purpose
            fc_series = pd.Series(fc, index=index_of_fc)
            lower_series = pd.Series(confint[:, 0], index=index_of_fc)
            upper_series = pd.Series(confint[:, 1], index=index_of_fc)

            # Plot
            plt.plot(df.value)
            plt.plot(fc_series, color='darkgreen')
            plt.fill_between(lower_series.index, 
                            lower_series, 
                            upper_series, 
                            color='k', alpha=.15)

            plt.title("Simple forecast")
            plt.show()

        except Exception as e:

            print("[FAILED] Caused by: {}".format(e))

if __name__ == "__main__":
    flow = operationsArima()
    flow.ForecastingWithArima() # Init script
