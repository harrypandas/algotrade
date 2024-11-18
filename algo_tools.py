import wrds
from datetime import date
from fredapi import Fred
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm

class APIConnector:
    def __init__(self, wrds_username=None, fred_api_key=None):
        self.db = None
        self.fred = None
        if wrds_username:
            self.db = wrds.Connection(wrds_username=wrds_username)
        if fred_api_key:
            self.fred = Fred(api_key=fred_api_key)

    def get_wrds_connection(self):
        return self.db

    def get_fred_connection(self):
        return self.fred
    
class algo_tools:
    def __init__(self, api_connector=None):
        self.api_connector = api_connector
        if api_connector:
            self.db = api_connector.get_wrds_connection()
            self.fred = api_connector.get_fred_connection()

    def get_crsp_permno_ticker(self, ticker):
        query = f"""
            SELECT ticker, comnam, permno, st_date, end_date 
            FROM crsp.stocknames
            WHERE ticker = '{ticker}'
        """
        df = self.db.raw_sql(
            query,
            date_cols=['date']
        )
        return df

    def get_crsp_daily_stock_returns(self, permco, start_date='2000-01-01', end_date=date.today().strftime('%Y-%m-%d')):
        query = f"""
            SELECT * FROM crsp.dsf
            WHERE date >= '{start_date}' AND date <= '{end_date}' AND permno={permco}
        """
        df = self.db.raw_sql(
            query,
            date_cols=['date']
        )
        df['cumret'] = (1 + df['ret']).cumprod() - 1
        df['cumretx'] = (1 + df['retx']).cumprod() - 1
        return df

    def get_crsp_monthly_stock_returns(self, permco, start_date='2000-01-01', end_date=date.today().strftime('%Y-%m-%d')):
        query = f"""
            SELECT * FROM crsp.msf
            WHERE date >= '{start_date}' AND date <= '{end_date}' AND permno={permco}
        """
        df = self.db.raw_sql(
            query,
            date_cols=['date']
        )
        df['cumret'] = (1 + df['ret']).cumprod() - 1
        df['cumretx'] = (1 + df['retx']).cumprod() - 1
        return df

    def get_tfz_ylds(self, start_date='2000-01-01', end_date=date.today().strftime('%Y-%m-%d')):
        query = f"""
            SELECT * FROM crsp.tfz_dly
            WHERE caldt >= '{start_date}' AND caldt <= '{end_date}'
        """
        df = self.db.raw_sql(
            query,
            date_cols=['date']
        )
        return df

    def get_yfinance_returns(self, ticker, start_date='2000-01-01', end_date=date.today().strftime('%Y-%m-%d')):
        df = yf.download(ticker, start=start_date, end=end_date)
        # df['ret'] = df['Adj Close'].pct_change()
        return df

    def get_fred_data(self, series='DGS10'):
        df = self.fred.get_series(series)
        df = df.reset_index(name='met')
        df.columns = ['date', 'met']
        return df

    @staticmethod
    def write_to_parquet(df, filename):
        df.to_parquet(filename)

    @staticmethod
    def calculate_geometric_mean_returns(returns):
        geometric_mean_returns = (1 + returns).prod() ** (1 / len(returns)) - 1
        return geometric_mean_returns

    @staticmethod
    def calculate_annualized_gmreturns(geometric_mean_returns, annual_factor=252):
        annualized_returns = ((1 + geometric_mean_returns) ** (annual_factor) - 1)
        return annualized_returns
    
    def reg_met(factor,fund_ret,constant = True):
        if constant:
            X = sm.tools.add_constant(factor)
        else:
            X = factor
        y=fund_ret
        model = sm.OLS(y,X,missing='drop').fit()
        
        if constant:
            beta = model.params[1:]
            alpha = round(float(model.params['const']),6)
            
        else:
            beta = model.params
        treynor_ratio = ((fund_ret.values).mean()*12)/beta[0]
        tracking_error = (model.resid.std()*np.sqrt(12))
        if constant:        
            information_ratio = model.params[0]*12/tracking_error
        r_squared = model.rsquared
        if constant:
            return (beta,treynor_ratio,information_ratio,alpha,r_squared,tracking_error)
        else:
            return (beta,treynor_ratio,r_squared,tracking_error)
    
    def calc_risk_metrics(data, benchmark, as_df=False, var=0.05, adj=12):
        """
        Calculate risk metrics for a DataFrame of assets.

        Args:
            data (pd.DataFrame): DataFrame of asset returns.
            as_df (bool, optional): Return a DF or a dict. Defaults to False.
            adj (int, optional): Annualizatin. Defaults to 12.
            var (float, optional): VaR level. Defaults to 0.05.

        Returns:
            Union[dict, DataFrame]: Dict or DataFrame of risk metrics.
        """
        summary = dict()
        summary["Mean"] = data.mean() * adj
        summary["Volatility"] = data.std() * np.sqrt(adj)
        summary["Sharpe Ratio"] = summary["Mean"] / summary["Volatility"]
        summary["Sortino Ratio"] = summary["Mean"] / (data[data < 0].std() * np.sqrt(adj))
        summary["Skewness"] = data.skew()   
        summary["Excess Kurtosis"] = data.kurtosis()
        summary[f"VaR ({var})"] = data.quantile(var, axis=0)
        summary[f"CVaR ({var})"] = data[data <= data.quantile(var, axis=0)].mean()
        summary["Min"] = data.min()
        summary["Max"] = data.max()
        excess_returns = pd.concat([benchmark, data], axis=1).dropna()
        excess_returns = (excess_returns.iloc[:, 1] - excess_returns.iloc[:, 0]).values.reshape(-1,1)
        mean_excess_returns = excess_returns.mean() * adj
        summary['Tracking Error'] = excess_returns.std() * np.sqrt(adj)
        summary['Information Ratio'] = mean_excess_returns/summary['Tracking Error']


        wealth_index = 1000 * (1 + data).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks

        summary["Max Drawdown"] = drawdowns.min()

        # summary["Bottom"] = drawdowns.idxmin()
        # summary["Peak"] = previous_peaks.idxmax()

        # recovery_date = []
        # for col in wealth_index.columns:
        #     prev_max = previous_peaks[col][: drawdowns[col].idxmin()].max()
        #     recovery_wealth = pd.DataFrame([wealth_index[col][drawdowns[col].idxmin() :]]).T
        #     recovery_date.append(
        #         recovery_wealth[recovery_wealth[col] >= prev_max].index.min()
        #     )

        # summary["Recovery"] = ["-" if pd.isnull(i) else i for i in recovery_date]

        # summary["Duration (days)"] = [
        #     (i - j).days if i != "-" else "-"
        #     for i, j in zip(summary["Recovery"], summary["Bottom"])
        # ]

        return pd.DataFrame(summary, index=data.columns) if as_df else summary
    

    def calc_return_metrics(data, as_df=False, adj=12):
        """
        Calculate return metrics for a DataFrame of assets.

        Args:
            data (pd.DataFrame): DataFrame of asset returns.
            as_df (bool, optional): Return a DF or a dict. Defaults to False (return a dict).
            adj (int, optional): Annualization. Defaults to 12.

        Returns:
            Union[dict, DataFrame]: Dict or DataFrame of return metrics.
        """
        summary = dict()
        summary["Annualized Return"] = data.mean() * adj
        summary["Annualized Volatility"] = data.std() * np.sqrt(adj)
        summary["Annualized Sharpe Ratio"] = (
            summary["Annualized Return"] / summary["Annualized Volatility"]
        )
        summary["Annualized Sortino Ratio"] = summary["Annualized Return"] / (
            data[data < 0].std() * np.sqrt(adj)
        )
        return pd.DataFrame(summary, index=data.columns) if as_df else summary
    

    def regression_based_performance(factor,fund_ret,rf,constant = True):
        """ 
            Returns the Regression based performance Stats for given set of returns and factors
            Inputs:
                factor - Dataframe containing monthly returns of the regressors
                fund_ret - Dataframe containing monthly excess returns of the regressand fund
                rf - Monthly risk free rate of return
            Output:
                summary_stats - (Beta of regression, treynor ratio, information ratio, alpha). 
        """
        if constant:
            X = sm.tools.add_constant(factor)
        else:
            X = factor
        y=fund_ret
        model = sm.OLS(y,X,missing='drop').fit()
        
        if constant:
            beta = model.params[1:]
            alpha = round(float(model.params['const']),6) *12

            
        else:
            beta = model.params
        treynor_ratio = ((fund_ret - rf).mean()*12)/beta[0]
        tracking_error = (model.resid.std()*np.sqrt(12))
        if constant:        
            information_ratio = model.params[0]*12/tracking_error
        r_squared = model.rsquared
        if constant:
            return (beta,treynor_ratio,information_ratio,alpha,r_squared,tracking_error,model.resid)
        else:
            return (beta,treynor_ratio,r_squared,tracking_error,model.resid)
        

    def tangency_portfolio_rfr(asset_return,cov_matrix):
        """ 
            Returns the tangency portfolio weights in a (1 x n) vector when a riskless assset is available
            Inputs: 
                asset_return - Excess return over the risk free rate for each asset (n x 1) Vector
                cov_matrix = nxn covariance matrix for the assets
        """
        asset_cov = np.array(cov_matrix)
        inverted_cov= np.linalg.inv(asset_cov)
        one_vector = np.ones(len(cov_matrix.index))
        
        den = (one_vector @ inverted_cov) @ (asset_return)
        num =  inverted_cov @ asset_return
        return (1/den) * num