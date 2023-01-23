"""strategy_old.py"""

import pandas as pd
import numpy as np

import logging

# logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s : %(module)s : %(funcName)s : %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class Strategy:
    """Definition of Strategy"""

    def __init__(self, instrument_price_df: pd.DataFrame = None, calendar=None, strategy_filter=None,
                 review_filter=None, weight=None, annual_fee: float = 0.0, annual_factor_fee: int = 365,
                 transaction_costs: {dict, float}=None, initial_value: float = 100.0):
        """
        Strategy is defined by a calendar, filter and weight as well as index calculation parameters such as costs
        :param instrument_price_df: DataFrame of prices for each components
        :param calendar: DataFrame or a BacktestCalendar object
        :param strategy_filter: DataFrame, FilterHandler or an instance of a subclass of BacktestFilter
        :param review_filter: DataFrame, FilterHandler or an instance of a subclass of BacktestFilter
        :param weight: DataFrame or an instance of a subclass of Weight
        :param annual_fee: float
        :param annual_factor_fee: int
        :param transaction_costs: float or dict with instrument names as keys and transaction costs as values
        :param initial_value: float
        """
        logger.info('initialize an instance of a Strategy object')

        # attributes used for the index construction
        self.instrument_price_df = instrument_price_df
        self.calendar = calendar
        self.strategy_filter = strategy_filter
        self.review_filter = review_filter
        self.weight = weight

        # attributes used for the calculation
        self.annual_fee = annual_fee
        self.annual_factor_fee = annual_factor_fee
        self.transaction_costs = transaction_costs
        self.initial_value = initial_value

        # read only data attributes
        self._calendar_df = None
        self._filter_df = None
        self._weight_df = None

    def run_backtest(self, set_calendars: bool = True, run_filters: bool = True, calculate_weights: bool = True,
                     strategy_calculation_method: str = 'basic')->pd.DataFrame:
        """
        Runs the back test of the strategy and returns a DataFrame with the results
        :param set_calendars: bool
        :param run_filters: bool
        :param calculate_weights: bool
        :param strategy_calculation_method: str 'basic' (assuming daily rebalance of all weights) or 'advanced'
        :return: DataFrame
        """
        logger.info('running back test of strategy')
        # setup the various building blocks of the strategy (if applicable)
        if set_calendars:
            self.set_calendars()

        if run_filters:
            self.run_filters()

        if calculate_weights:
            self.calculate_weight()

        if self.weight is None:
            raise ValueError("the weights are not defined. specify weight attribute and set calculate_weights to True "
                             "in run_backtest method")

        # adjust the prices and weights
        price_df, adj_weight_df = self._clean_price_and_weight_df()

        # (optional) calculate overlay (volatility target, stop-loss, beta hedge etc.)
        # TODO add overlays (daily price, weight and return can be changed by the overlays like beta hedge or smoothing)

        # calculate the index
        logger.debug(f"calculating strategy ('{strategy_calculation_method}' method)")
        if strategy_calculation_method == 'advanced':
            return advanced_strategy_calculation(price_df=price_df, weight_df=adj_weight_df, initial_value=self.initial_value,
                                                 transaction_costs=self.transaction_costs, fee_pa=self.annual_fee,
                                                 annual_factor_fee=self.annual_factor_fee)
        elif strategy_calculation_method == 'basic':
            # assumes the weights are rebalanced daily
            daily_weight_df = adj_weight_df.reindex_like(price_df, method='ffill')  # make the weight_df daily
            daily_return_df = price_df.fillna(method='ffill').pct_change()

            # calculate the costs related to fees and transaction costs (if any)
            index_costs_s = self._get_strategy_costs_series(daily_weight_df=daily_weight_df)

            strategy_df = pd.DataFrame(
                data=(1 + (daily_return_df * daily_weight_df.shift(1).values).sum(
                    axis=1) - index_costs_s.values).cumprod(),
                index=daily_return_df.index,
                columns=['strategy']
            )
            strategy_df *= self.initial_value
            return strategy_df
        else:
            raise ValueError(f"'{strategy_calculation_method}' is not a recognized strategy calculation method")

    def _clean_price_and_weight_df(self)->tuple:
        """
        Returns a price and weight DataFrame as a tuple after having removed the columns that only has 0 weights for all dates.
        Only keep the daily prices that starts after (and including) the first rebalance date and remove the dates where
        all prices are nan
        :return: tuple (price: DataFrame, weight: DataFrame)
        """
        # clean up the weights (remove columns with all rows 0)
        weight_df = self.weight_df.fillna(0).copy()
        adj_weight_df = weight_df.loc[:, (weight_df != 0).any(axis=0)]
        # only use the prices for the instruments with existing weights and start from the first weight observation
        price_df = self.instrument_price_df.loc[adj_weight_df.index[0]:, adj_weight_df.columns].copy()
        price_df = price_df.loc[~price_df.isnull().all(axis=1), :]  # remove the rows where all prices are nan
        return price_df, weight_df

    def _get_strategy_costs_series(self, daily_weight_df: pd.DataFrame)->pd.Series:
        """
        Returns a Series of the sum of the rolling fees and the transaction costs
        :param daily_weight_df: DataFrame
        :return: Series
        """
        # calculate the rolling fees (if any)
        index_costs_s = daily_weight_df.index.to_series().diff().dt.days.fillna(
            0) / self.annual_factor_fee * self.annual_fee

        # calculate the transaction costs (if any)
        if isinstance(self.transaction_costs, dict):
            t_cost_s = pd.Series(data=self.transaction_costs)
            t_cost_s = t_cost_s[daily_weight_df.columns].fillna(0)
        else:
            t_cost_s = self.transaction_costs

        # add the sum of the absolute changes in weights multiplied by the transaction cost
        index_costs_s += (daily_weight_df.fillna(0).diff().abs().fillna(0) * t_cost_s).sum(axis=1)
        return index_costs_s

    def set_calendars(self)->None:
        """
        Sets the _calendar_df attribute using the calendar attribute handling the cases when calendar is a DataFrame
        or an instance BackTestCalendar object
        :return: None
        """
        logger.info('setup calendars')
        # it it is a DataFrame make sure the required columns exists
        if isinstance(self.calendar, pd.DataFrame):
            if {'rebalance_calendar', 'reweight_calendar', 'review_calendar'}.issubset(self.calendar.columns):
                logger.debug('calendar was given as a DataFrame')
                self._calendar_df = self.calendar.copy()
            else:
                raise ValueError("the columns of calendar DataFrame needs to include 'rebalance_calendar', "
                                 "'reweight_calendar', 'review_calendar'")
        elif self.calendar is None:
            logger.debug('no calendar is specified so create a default one only containign the first date of the '
                         'specified instrument_price_df attribute')
            # by default set each calendar to the first observed date in the price DataFrame
            self._calendar_df = pd.DataFrame(
                {
                    'rebalance_calendar': [self.instrument_price_df.index[0]],
                    'reweight_calendar': [self.instrument_price_df.index[0]],
                    'review_calendar': [self.instrument_price_df.index[0]]
                }
            )
        else:
            # try to run the get_calendars method else raise an error
            try:
                self._calendar_df = self.calendar.get_calendars(as_dataframe=True)
            except AttributeError:
                raise TypeError("'calendar' needs to be a DataFrame or have a get_calendars function like an instance "
                                "\nof a BacktestCalendar")

    def run_filters(self)->None:
        """
        First run the review filter and adjust the calendar to each back test calendar. First run the review filter,
        then the reweight filter. Finally the reweight filters gets mapped to the rebalance calendar. The final filter
        is stored in _filter_df attribute
        :return: None
        """
        logger.info('run filters for the strategy')
        if self.review_filter is not None:
            review_filter_df = self._run_filter_on_calendar(_filter=self.review_filter, calendar_name='review_calendar')
        else:
            review_filter_df = None

        reweight_filter_df = self._run_filter_on_calendar(_filter=self.strategy_filter,
                                                          calendar_name='reweight_calendar',
                                                          previous_filter_df=review_filter_df)

        self._filter_df = reweight_filter_df
        return

    def _run_filter_on_calendar(self, _filter, calendar_name: str, previous_filter_df: pd.DataFrame = None)->pd.DataFrame:
        """
        Run a filter and forward fill the filter values on a specified calendar
        :param _filter: None, DataFrame or an instance of a class with a run() method like FilterHandler and subclasses
        of BackTestFilter
        :param calendar_name: str
        :param previous_filter_df: None or DataFrame
        :return:
        """
        logger.debug(f"run filter on '{calendar_name}'")
        if self.calendar_df is None:
            raise ValueError("_calendar_df is None: run set_calendars")

        if _filter is None:
            logger.debug(f"no filter specified so return a default one as a DataFrame with value 1 if price exists, "
                         f"else nan")
            # DataFrame with 1 for numeric values, else nan
            return pd.DataFrame(
                np.where(
                    self.instrument_price_df.isnull(), np.nan, 1
                ),
                index=self.instrument_price_df.index,
                columns=self.instrument_price_df.columns
            )

        if isinstance(_filter, pd.DataFrame):
            logger.debug(f"filter given as a DataFrame")
            filter_df = _filter
        else:
            try:
                # TODO ugly solution (distingushing when _filter is a Filter object or FilterHandler
                if 'previous_filter_df' in dir(_filter):  # an instance of a subclass of Filter
                    logger.debug(f"run filter using a subclass of Filter")
                    _filter.previous_filter_df = previous_filter_df
                    filter_df = _filter.run()
                else:
                    logger.debug(f"run filter using a FilerHandler")
                    filter_df = _filter.run(previous_filter_df=previous_filter_df)
            except AttributeError:
                raise TypeError("'filter' needs to be a DataFrame or have a run function like an instance "
                                "\nof a FilterHandler or a subclass of BacktestFilter")

        # forward fill the result with respect to the specified calendar
        return self._ffill_values_on_calendar(df=filter_df, calendar_name=calendar_name)

    def _ffill_values_on_calendar(self, df: pd.DataFrame, calendar_name: str)->pd.DataFrame:
        """
        Forward fill the result with respect to the specified calendar
        :param df: DataFrame
        :param calendar_name: str
        :return: DataFrame
        """
        return df.reindex_like(pd.DataFrame(index=self.calendar_df[calendar_name], columns=df.columns), method="ffill")

    def calculate_weight(self):
        """
        Calculate the weights of the instruments that passed the filter (if any) and store the result in a weight
        attribute
        :return: None
        """
        logger.info("calculate the weights for the strategy")
        if isinstance(self.weight, pd.DataFrame):
            logger.debug("use specified DataFrame for the weights")
            self._weight_df = self.weight.copy()
        else:
            try:
                self.weight.eligibility_df = self.filter_df.copy()

                # forward fill the weight values from the reweight calendar to the rebalance calendar
                self._weight_df = self._ffill_values_on_calendar(df=self.weight.run(), calendar_name='rebalance_calendar')
            except AttributeError:
                logger.error("'weight' needs to be a DataFrame or an instance of a subclass of Weight with a run "
                             "method")
                raise TypeError("'weight' needs to be a DataFrame or an instance of a subclass of Weight with a run "
                                "method")

    @property  # calendar_df is read only
    def calendar_df(self):
        return self._calendar_df

    @property  # filter_df is read only
    def filter_df(self):
        return self._filter_df

    @property  # weight_df is read only
    def weight_df(self):
        return self._weight_df

    @property
    def transaction_costs(self):
        return self._transaction_costs

    @transaction_costs.setter
    def transaction_costs(self, transaction_costs: {dict, float}):
        if transaction_costs is None:
            self._transaction_costs = 0
        elif isinstance(transaction_costs, (float, int, dict)):
            self._transaction_costs = transaction_costs
        else:
            logger.error("transaction_costs should only be specified as float, dict or None")
            raise ValueError("transaction_costs should only be specified as float, dict or None")


def advanced_strategy_calculation(price_df: pd.DataFrame, weight_df: pd.DataFrame, initial_value: float = 100.0, transaction_costs: {float, dict}=0.0,
                                  fee_pa: float = None, annual_factor_fee: int = 365, output_details: bool = False):
    """
    Calculates a strategy that does not assume daily rebalancing of weights but instead takes into account transaction
    costs when rebalancing to a static weight that has drifted away from the initial weights. For example, a 50%/50%
    portfolio between two stocks will not stay at 50%/50% assuming they move with a beta at exactly 1.
    :param price_df: DataFrame
    :param weight_df: DataFrame
    :param initial_value: float
    :param transaction_costs: float, dict
    :param fee_pa: float
    :param annual_factor_fee: int
    :param output_details: bool, if False only return the time series, else include the calculation details
    :return: DataFrame
    """

    price_df, weight_df = _adjust_strategy_input(price_df=price_df, weight_df=weight_df)
    tickers = list(price_df.columns)

    # change the column names to the weight DataFrame and reset the index
    weight_col = [f"{c_name}_weight" for c_name in weight_df.columns]  # adds suffix to the tickers
    weight_df.reset_index(inplace=True)
    weight_df.columns = ['rbd'] + weight_col
    weight_df['rbd'] = pd.to_datetime(weight_df['rbd'])
    weight_df.index = weight_df['rbd']

    strategy_df = pd.merge_asof(price_df, weight_df, left_index=True, right_index=True)  # VLOOKUP with no exact match
    strategy_df.index.name = 'date'  # this is later used when merging the transaction costs
    strategy_df['is_rbd'] = strategy_df['rbd'].diff().dt.days > 0  # True if date is rebalance date, else False
    strategy_df.iloc[0, -1] = True  # first date is a rebalance date

    # add a column with last rebalance calendar adjusted in case the rebalance date does not exist in the daily calendar
    strategy_df = pd.merge_asof(strategy_df, pd.DataFrame({'adj_rbd': strategy_df.loc[strategy_df['is_rbd']].index},
                                                          index=strategy_df.loc[strategy_df['is_rbd']].index),
                                left_index=True, right_index=True)

    # replace the prices with the performance of each underlying since last rebalance date
    strategy_df[tickers] = strategy_df[tickers] / strategy_df.loc[strategy_df['is_rbd']][tickers].reindex_like(price_df, method='ffill').shift().values
    strategy_df.loc[strategy_df.index[0], tickers] = 1.0

    # calculate the transaction costs
    if transaction_costs:
        strategy_df = _add_transaction_costs(strategy_df=strategy_df, transaction_costs=transaction_costs, tickers=tickers)
    else:
        strategy_df['transaction_costs'] = 0

    # calculate the running fee
    if fee_pa:
        strategy_df['fees'] = (strategy_df.index - strategy_df['adj_rbd'].shift()).dt.days / annual_factor_fee * fee_pa
    else:
        strategy_df['fees'] = 0

    # calculate the strategy
    strategy_df['strategy_last_rbd'], strategy_df['strategy'] = None, None
    # strategy_df[['strategy_last_rbd', 'strategy']] = ''  # add two new columns
    strategy_df.iloc[0, [-2, -1]] = [initial_value, initial_value]
    weighted_net_ret_sum_arr = (((strategy_df[tickers] - 1) * strategy_df[weight_col].shift().values).sum(axis=1)
                                - strategy_df['transaction_costs'].values - strategy_df['fees'].values).to_numpy()
    strategy_df.reset_index(inplace=True)  # to be able to look up i instead of dates

    for i in range(1, len(strategy_df)):
        strategy_df.loc[i, 'strategy'] = strategy_df.loc[i - 1, 'strategy_last_rbd'] * (1 + weighted_net_ret_sum_arr[i])
        if strategy_df.loc[i, 'is_rbd']:
            strategy_df.loc[i, 'strategy_last_rbd'] = strategy_df.loc[i, 'strategy']
        else:
            strategy_df.loc[i, 'strategy_last_rbd'] = strategy_df.loc[i - 1, 'strategy_last_rbd']

    # set the index again to the dates and return the result
    strategy_df.set_index('date', inplace=True)
    if output_details:
        return strategy_df
    else:
        return strategy_df[['strategy']]


def _adjust_strategy_input(price_df: pd.DataFrame, weight_df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    Removes redundant rows in the price and weight DataFrame and checks so that the columns are the same
    :param price_df: DataFrame
    :param weight_df: DataFrame
    :return: (DataFrame, DataFrame)
    """

    # removes weight rows that are outside the price DataFrame
    weight_df = weight_df.loc[price_df.index[0]: price_df.index[-1], :].copy()

    # start the price from the first rebalance date
    price_df = price_df.loc[weight_df.index[0]:, :].copy()

    # check so that the columns are the same
    if set(weight_df.columns) != set(price_df.columns):
        raise ValueError("columns of 'price_df' and 'weight_df' are not the same")
    else:
        # make sure the order is the same
        weight_df = weight_df[price_df.columns]
    return price_df, weight_df


def _add_transaction_costs(strategy_df: pd.DataFrame, transaction_costs: {float, dict}, tickers: list) -> pd.DataFrame:
    """
    Adds column for transaction costs to the strategy DataFrame
    ABS(Weight[t] - Weight[last RBD] * Price[t] / Price[last RBD]) * Transaction cost
    Notice that even if the weights don't change between rebalance dates, the strategy will still incur transaction
    costs since the actual weights will have deviate from the target weights
    :param strategy_df: DataFrame
    :param transaction_costs: float or dict
    :param tickers: list of str
    :return: DataFrame
    """

    weight_col = [f"{ticker}_weight" for ticker in tickers]
    weight_abs_delta_df = (
            strategy_df.loc[strategy_df['is_rbd']][weight_col]  # Weight[t]
            - strategy_df.loc[strategy_df['is_rbd']][weight_col].shift()  # Weight[last RBD]
            * strategy_df.loc[strategy_df['is_rbd']][tickers].values     # Price[t] / Price[last RBD]
                           ).abs()

    if isinstance(transaction_costs, dict):
        # capital letters for the keys i.e. tickers and ignore keys that don't exists as tickers in the DataFrame
        transaction_costs = {f"{col.upper()}_weight": val for col, val in transaction_costs.items()
                             if col.upper() in tickers}
        weight_abs_delta_df[list(transaction_costs.keys())] *= pd.Series(transaction_costs)
    else:
        # assumes the transaction_costs is a float
        weight_abs_delta_df *= transaction_costs

    transaction_cost_df = pd.DataFrame(weight_abs_delta_df.sum(axis=1).rename('transaction_costs'))
    transaction_cost_df.index.name = 'date'
    strategy_df = pd.merge(strategy_df, transaction_cost_df, how='outer', on='date')
    strategy_df['transaction_costs'] = strategy_df['transaction_costs'].fillna(0)
    return strategy_df






