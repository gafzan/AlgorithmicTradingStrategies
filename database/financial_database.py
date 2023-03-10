"""
financial_database.py
"""
from sqlalchemy import create_engine, and_, func
from sqlalchemy.orm import sessionmaker
import logging
import yfinance
from datetime import date, datetime, timedelta
import pandas as pd
from pandas.tseries.offsets import BDay
import numpy as np
from tkinter import filedialog, Tk
import xlrd
from dateutil.parser import parse as str_to_datetime

# my own modules
from database.models_db import Base, Underlying, OpenPrice, HighPrice, LowPrice, ClosePrice, Volume, Dividend
from general_tools import capital_letter_no_blanks, list_grouper, extend_dict, reverse_dict, progression_bar
from dataframe_tools import select_rows_from_dataframe_based_on_sub_calendar
from database.config_database import __MY_DATABASE_NAME__, __DATABASE_FEED_EXCEL_FILES_FOLDER__
from excel_tools import load_df
from database.bloomberg import BloombergConnection
from financial_analysis import financial_time_series_functions as fin_ts

# Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s : %(module)s : %(funcName)s : %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class FinancialDatabase:
    """Class definition for FinancialDatabase.
    This class allows for reading and deleting data but not for creating or updating data."""

    def __init__(self, database_name: str, database_echo=False):
        self._database_name = database_name
        self._database_echo = database_echo
        engine = create_engine(self.database_name, echo=self.database_echo)
        Base.metadata.create_all(engine)  # create the database tables (these are empty at this stage)
        Session = sessionmaker(bind=engine)  # ORM's 'handle' to the database (bound to the engine object)
        self._session = Session()  # whenever you need to communicate with the database you instantiate a Session
        self._delete_dates_between_start_end = True

    # ------------------------------------------------------------------------------------------------------------------
    # methods for Underlying table

    def underlying_exist(self, ticker: str) -> bool:
        """Assumes that ticker is a string. Returns True if there is an Underlying row that is represented by the
        ticker, else returns False."""
        ticker = self.reformat_tickers(ticker)
        logger.debug("Checking for existence of the ticker '{}'.".format(ticker))
        return self.session.query(Underlying).filter(Underlying.ticker == ticker).count() > 0

    def delete_underlying(self, tickers: {str, list}) -> None:
        """Assumes that tickers is either a string or a list of strings. Deletes all Underlying rows corresponding to
        the ticker(s). This will also remove all rows from tables that are subclasses to Underlying."""
        tickers = self.reformat_tickers(tickers, convert_to_list=True)
        logger.debug("Trying to delete {} ticker(s) from the database.\nTicker(s): %s".format(len(tickers)) % ', '.join(tickers))
        tickers_that_exists = [ticker for ticker in tickers if self.underlying_exist(ticker)]
        if len(tickers_that_exists) < len(tickers):
            logger.warning('{} ticker(s) could not be deleted since they do not exist in the database.'.format(len(tickers) - len(tickers_that_exists)))
        for ticker in tickers_that_exists:
            query_underlying = self.session.query(Underlying).filter(Underlying.ticker == ticker).first()
            logger.debug('Delete {} from the database.'.format(ticker))
            self.session.delete(query_underlying)
        self.session.commit()
        return

    def _delete_open_high_low_close_volume_dividend_data(self, table, ticker_list: list, date_list: list) -> None:
        """Assumes that tickers is a list of strings and start_date and end_date are of type datetime. Deletes all rows
        in OpenPrice, HighPrice, LowPrice, ClosePrice, Volume and Dividend table for the given tickers from start_date
        to end_date."""
        if len(date_list) == 0:
            logger.debug('No {} data to delete for {} ticker(s).'.format(table.__tablename__, len(ticker_list)))
            return
        else:
            pass

        underlying_id_list = self.get_ticker_underlying_attribute_dict(ticker_list, Underlying.id).values()

        try:
            date_variable_in_table = table.obs_date
        except AttributeError:
            # in case of Dividend table use 'ex-dividend date'
            date_variable_in_table = table.ex_div_date

        # make sure the dates are of type datetime
        try:
            date_list = [datetime(year=date_.year, month=date_.month, day=date_.day) for date_ in date_list]
        except AttributeError:
            date_list = [pd.to_datetime(date_) for date_ in date_list]

        if self._delete_dates_between_start_end:
            start_date = min(date_list)
            end_date = max(date_list)
            logger.debug("Deleting {} data for {} ticker(s)".format(table.__tablename__, len(ticker_list))
                         + logger_time_interval_message(start_date, end_date) +
                         '\nTicker(s): %s' % ', '.join(ticker_list))
            dates_eligible_for_deletion = and_(date_variable_in_table >= start_date, date_variable_in_table <= end_date)
            for underlying_id_sub_list in list_grouper(underlying_id_list, 500):
                self._session.query(table) \
                    .filter(and_(table.underlying_id.in_(underlying_id_sub_list), dates_eligible_for_deletion)) \
                    .delete(synchronize_session=False)
        else:
            logger.debug("Deleting {} data for {} ticker(s) over {} observation dates."
                         .format(table.__tablename__, len(ticker_list), len(date_list)) +
                         '\nTicker(s): %s' % ', '.join(ticker_list))
            # for each underlying id, query the data that exists between the start and end date and remove the content
            for underlying_id_sub_list in list_grouper(underlying_id_list, 500):
                for date_sub_list in list_grouper(date_list, 100):
                    self._session.query(table) \
                        .filter(and_(table.underlying_id.in_(underlying_id_sub_list),
                                     date_variable_in_table.in_(date_sub_list))) \
                        .delete(synchronize_session=False)
        self.session.commit()

    def get_ticker(self, underlying_attribute_dict: dict = None) -> list:
        """Assume that underlying_attribute_dict is a dictionary with Underlying.attribute_name (e.g. Underlying.sector)
        as ???key??? and attribute value (e.g. ???INDUSTRIALS???) as ???value???. Returns a list of tickers (each a string) who
        shares the attributes.
        To find the union of tickers with two values of attributes (e.g. all tickers in ???INDUSTRIALS??? and ???ENERGY???)
        simply have the 'key' (Underlying.sector) point to a list ([???INDUSTRIALS???, ???ENERGY???]).
        The resulting dictionary will hence look like {Underlying.sector:  [???INDUSTRIALS???, ???ENERGY???]}.
        Adding several keys will mean that you are taking the intersection between the attributes. E.g. inputting the
        dictionary {Underlying.sector:  [???INDUSTRIALS???, ???ENERGY???], Underlying.currency: ???JPY???} will lead to the method
        returning tickers for stocks in the Industrial and Energy sector that are all denominated in JPY."""
        if underlying_attribute_dict is None:
            underlying_attribute_dict = {}  # selection will be based on no attributes i.e. select all tickers
        underlying_attribute_list = underlying_attribute_dict.keys()
        logger_message = "Filtering tickers based on: \n" + "\n".join(["{} = {}".format(key, value)
                                                                       for key, value in underlying_attribute_dict.items()])
        logger.info(logger_message)

        query_ticker = self.session.query(Underlying.ticker)
        for underlying_attribute in underlying_attribute_list:
            underlying_attribute_value = underlying_attribute_dict[underlying_attribute]  # e.g. 'ENERGY' if sector
            if not isinstance(underlying_attribute_value, list):
                underlying_attribute_value = [underlying_attribute_value]  # string -> [string]
            if underlying_attribute in [Underlying.ticker]:
                # capital letters with blanks
                underlying_attribute_value = [attribute.upper() for attribute in underlying_attribute_value.copy()]
            elif underlying_attribute not in [Underlying.has_dividend_history, Underlying.latest_observation_date,
                                              Underlying.latest_observation_date_with_values, Underlying.oldest_observation_date,
                                              Underlying.first_ex_div_date]:
                # capital letters without blanks (replaced by '_')
                underlying_attribute_value = capital_letter_no_blanks(underlying_attribute_value)
            else:
                pass
            query_ticker = query_ticker.filter(underlying_attribute.in_(underlying_attribute_value))
        query_ticker.order_by(Underlying.ticker)
        ticker_list = [tup[0] for tup in query_ticker]  # extract the ticker string from the result
        logger.info("{} ticker(s) selected.".format(len(ticker_list)))
        return ticker_list

    def get_ticker_underlying_attribute_dict(self, tickers: {str, list}, underlying_attribute) -> dict:
        """Assumes that ticker is a string or a list of strings and attribute is of type
        sqlalchemy.orm.attributes.InstrumentedAttribute (e.g. Underlying.sector will return a dictionary like
        {'ticker 1': 'sector A', 'ticker 2': 'sector B' ,...}).
        Returns a dictionary with tickers as keys and the specific attribute as values."""
        tickers = self.reformat_tickers(tickers, convert_to_list=True)
        ticker_attribute_dict = {}  # initializing the dictionary

        # to make the requests smaller, we need to split the ticker list into sub list
        for ticker_sub_list in list_grouper(tickers, 500):
            query_ticker_attribute = self.session.query(Underlying.ticker, underlying_attribute) \
                .filter(
                Underlying.ticker.in_(ticker_sub_list))\
                .order_by(Underlying.ticker)
            ticker_attribute_dict = extend_dict(ticker_attribute_dict, dict(query_ticker_attribute))
        return ticker_attribute_dict

    def get_underlying_data(self, tickers: {str, list}, attribute: {str, list, tuple}) -> pd.DataFrame:
        """
        Returns a DataFrame with the underlying attributes for each ticker in the list.
        :param tickers: string or iterable of strings with the tickers
        :param attribute: string or iterable of strings with the names of the attributes e.g. 'sector'
        :return: pd.DataFrame
        """
        result_df = None
        tickers = self.reformat_tickers(tickers, convert_to_list=True)
        if isinstance(attribute, str):
            attribute = [attribute]
        for attr in attribute:
            # for all attributes, store the value next to the ticker in a DataFrame
            ticker_attribute_dict = self.get_ticker_underlying_attribute_dict(tickers, attr)
            result_sub_df = pd.DataFrame(ticker_attribute_dict, index=range(1))
            result_sub_df = result_sub_df.unstack().reset_index()
            result_sub_df.drop('level_1', inplace=True, axis=1)  # remove unnecessary column
            result_sub_df.columns = ['ticker', attr]
            if result_df is None:
                result_df = result_sub_df
            else:
                # concatenate the DataFrame
                result_df = pd.merge(result_df, result_sub_df, on='ticker')
        result_df.set_index('ticker', inplace=True)
        return result_df

    def _update_obs_date_and_dividend_history_status(self,  tickers: list) -> None:
        """Assumes tickers is a list of strings. For each ticker, method assign to the Underlying table 1) latest
        observation date, 2) latest observation date with value, 3) oldest observation date and 4) first ex-dividend
        date (if any)."""

        logger.debug('Updating oldest and latest observation date and first ex-dividend date for {} ticker(s).'.format(len(tickers)))
        underlying_id_list = list(self.get_ticker_underlying_attribute_dict(tickers, Underlying.id).values())
        self._update_dividend_info(underlying_id_list)
        self._update_obs_date(underlying_id_list)
        self.session.commit()
        return

    def _update_dividend_info(self, underlying_id_list: list) -> None:
        """
        For all underlyings represented by the underlying id in the given list, set the oldest ex-dividend date in the
        database.
        :param underlying_id_list: list of int
        :return: None
        """
        logger.debug('Updating oldest ex-dividend date and dividend paying status.')

        # initialize the list that will store the results from the query
        oldest_ex_div_date = []

        for underlying_id_sub_list in list_grouper(underlying_id_list, 500):
            # query the oldest ex-dividend date
            sub_query_oldest_div_date = self.session.query(
                Dividend.underlying_id,
                func.min(Dividend.ex_div_date)
            ).group_by(
                Dividend.underlying_id
            ).filter(
                Dividend.underlying_id.in_(underlying_id_sub_list)
            ).all()
            oldest_ex_div_date.extend(sub_query_oldest_div_date)

        # create a dictionary containing a tuple (keys = underlying id, values = oldest ex-dividend date)
        underlying_id_oldest_ex_div_date_dict = {tup[0]: tup[1] for tup in oldest_ex_div_date}

        # update the database
        for underlying_id in list(underlying_id_oldest_ex_div_date_dict.keys()):
            self.session.query(
                Underlying
            ).filter(
                Underlying.id == underlying_id
            ).update(
                {'first_ex_div_date': underlying_id_oldest_ex_div_date_dict[underlying_id],
                 'has_dividend_history': True}
            )
        logger.debug('Done with updating ex dividend dates.')
        return

    def _update_obs_date(self, underlying_id_list: list) -> None:
        """
        For all underlyings represented by the underlying id in the given list, set the oldest, latest and latest with
        value observation date.
        database.
        :param underlying_id_list: list of int
        :return: None
        """
        logger.debug('Updating oldest and latest observation date.')

        # initialize the list that will store the results from the queries
        max_min_obs_date = []
        max_obs_date_with_values = []

        for underlying_id_sub_list in list_grouper(underlying_id_list, 500):
            # first query the oldest and latest observation date
            sub_query_max_min_obs_date = self.session.query(
                ClosePrice.underlying_id,
                func.min(ClosePrice.obs_date),
                func.max(ClosePrice.obs_date)
            ).group_by(
                ClosePrice.underlying_id
            ).filter(
                ClosePrice.underlying_id.in_(underlying_id_sub_list)
            ).all()

            # also query the observation dates with recorded close price (i.e. not nan)
            sub_query_max_obs_date_with_value = self.session.query(
                ClosePrice.underlying_id,
                func.max(ClosePrice.obs_date)
            ).group_by(
                ClosePrice.underlying_id
            ).filter(
                ClosePrice.underlying_id.in_(underlying_id_sub_list),
                ClosePrice.close_quote.isnot(None)
            ).all()
            max_min_obs_date.extend(sub_query_max_min_obs_date)
            max_obs_date_with_values.extend(sub_query_max_obs_date_with_value)

        # create a dictionary containing a tuple: keys = underlying id, values = (oldest obs. date, latest obs. date,
        # latest obs. date with value)
        underlying_id_max_min_obs_date_dict = {tup[0]: (tup[1], tup[2]) for tup in max_min_obs_date}
        underlying_id_max_obs_date_with_value_dict = {tup[0]: tup[1] for tup in max_obs_date_with_values}
        underlying_obs_date_dict = {underlying_id: underlying_id_max_min_obs_date_dict[underlying_id]
                                                   + (underlying_id_max_obs_date_with_value_dict[underlying_id], )
                                    for underlying_id in list(underlying_id_max_min_obs_date_dict.keys())}

        # update the database
        for underlying_id in list(underlying_obs_date_dict.keys()):
            self.session.query(
                Underlying
            ).filter(
                Underlying.id == underlying_id
            ).update(
                {'latest_observation_date': underlying_obs_date_dict[underlying_id][1],
                 'latest_observation_date_with_values': underlying_obs_date_dict[underlying_id][2],
                 'oldest_observation_date': underlying_obs_date_dict[underlying_id][0]}
            )
        logger.debug('Done with updating observation dates.')
        return

    # ------------------------------------------------------------------------------------------------------------------
    # methods for OpenPrice, HighPrice, LowPrice, ClosePrice, Volume and Dividend tables

    def _get_open_high_low_close_volume_dividend_df(self, table, tickers: {str, list}, start_date: {date, datetime},
                                                    end_date: {date, datetime}, currency: {str, None}, ffill_na: bool,
                                                    bfill_na: bool, drop_na: bool)->pd.DataFrame:
        tickers, start_date, end_date = self._input_check_before_getting_ohlc_volume(tickers, start_date, end_date)
        logger.info('Get {} data for {} ticker(s)'.format(table.__tablename__, len(tickers))
                     + logger_time_interval_message(start_date, end_date))

        # need to add an extra day otherwise the 'between' function below does not capture the end date
        end_date = end_date + timedelta(1)

        # dictionary that holds the name of the value column
        value_column_name_dict = {OpenPrice: 'open_quote', HighPrice: 'high_quote', LowPrice: 'low_quote',
                                  ClosePrice: 'close_quote', Volume: 'volume_quote', Dividend: 'dividend_amount'}

        ticker_underlying_id_dict = self.get_ticker_underlying_attribute_dict(tickers, Underlying.id)
        underlying_id_list = ticker_underlying_id_dict.values()
        result_df = None
        # for each underlying id, query data from the requested table
        for underlying_id_sub_list in list_grouper(underlying_id_list, 500):
            if table == Dividend:
                table_date = table.ex_div_date
            else:
                table_date = table.obs_date
            # query date, value and underlying.id for the specific tickers and between the specific start and end date
            query_date_value_id = self.session.query(table_date,
                                                     value_column_name_dict[table],
                                                     table.underlying_id)\
                .filter(
                and_(table.underlying_id.in_(underlying_id_sub_list),
                     table_date.between(start_date, end_date))
            )
            sub_df = pd.read_sql_query(query_date_value_id.statement, self.session.get_bind())
            if result_df is None:  # first sub list, no need to concatenate the DataFrame
                result_df = sub_df
            else:
                result_df = pd.concat([result_df, sub_df], ignore_index=True)

        # pivot the DataFrame s.t. dates are the index and Underlying.id are the column headers
        result_pivoted_df = result_df.pivot(index=result_df.columns[0], columns=result_df.columns[2],
                                            values=value_column_name_dict[table])
        # change the column names from Underlying.id to ticker
        underlying_id_ticker_dict = reverse_dict(ticker_underlying_id_dict)
        column_names = [underlying_id_ticker_dict[underlying_id] for underlying_id in
                        result_pivoted_df.columns]
        result_pivoted_df.columns = column_names

        # handle tickers that has no data
        result_pivoted_df = self._handle_missing_data(tickers, result_pivoted_df)

        # if all columns are nan, remove the entire row
        result_pivoted_df.dropna(inplace=True, how='all')

        # currency convert if applicable
        if isinstance(currency, str) and table != Volume:
            result_pivoted_df = self._currency_convert_df(result_pivoted_df, currency)
        else:
            result_pivoted_df = result_pivoted_df
        # if all columns are nan, remove the entire row
        result_pivoted_df.dropna(inplace=True, how='all')

        # if applicable clean results
        if ffill_na:
            result_pivoted_df.fillna(method='ffill', inplace=True)

        if bfill_na:
            result_pivoted_df.fillna(method='bfill', inplace=True)

        if drop_na:
            result_pivoted_df.dropna(inplace=True)
        logger.info('Done with loading {}'.format(value_column_name_dict[table].replace('_', ' ') + 's'))
        return result_pivoted_df

    def _input_check_before_getting_ohlc_volume(self, tickers: {str, list}, start_date: {date, datetime}, end_date: {date, datetime}) -> tuple:
        """This method checks some of the inputs for _get_open_high_low_close_volume_dividend_df method. Returns a
        tuple with the inputs that have been adjusted if applicable."""
        # adjust inputs
        tickers = self.reformat_tickers(tickers, convert_to_list=True, sort=False)
        if len(tickers) == 0:
            raise TypeError('The ticker list was empty.')
        tickers_not_in_database = []
        for ticker in tickers:
            if not self.underlying_exist(ticker):
                tickers_not_in_database.append(ticker)
        if len(tickers_not_in_database) > 0:
            raise ValueError(
                "{} ticker(s) are missing from the database.\nTicker(s): %s".format(len(tickers_not_in_database)) % ", ".join(tickers_not_in_database))

        # adjust start date
        if start_date is None:
            # Pick the oldest observation date available
            start_date = min(
                self.get_ticker_underlying_attribute_dict(tickers, Underlying.oldest_observation_date).values())
        elif isinstance(start_date, str):
            start_date = str_to_datetime(start_date)

        if isinstance(start_date, datetime):
            start_date = start_date.date()

        # adjust end date
        if end_date is None:
            # Pick the latest observation date with data available
            end_date = max(self.get_ticker_underlying_attribute_dict(tickers,
                                                                     Underlying.latest_observation_date_with_values).values())
        elif isinstance(end_date, str):
            end_date = str_to_datetime(end_date)

        if isinstance(end_date, datetime):
            end_date = end_date.date()
        return tickers, start_date, end_date

    @staticmethod
    def _handle_missing_data(original_ticker_list: list, values_df: pd.DataFrame) -> pd.DataFrame:
        """Assume that original_ticker_list is a list of tickers that needs to be column names in values_df. If a ticker
        does not exist as a column name, insert a column with NaN as values. Returns a DataFrame."""
        missing_ticker_list = list(set(original_ticker_list).difference(list(values_df)))
        for missing_ticker in missing_ticker_list:
            values_df.insert(0, missing_ticker, np.nan)  # NaN for each date
        values_df = values_df[original_ticker_list]  # rearrange the column names
        return values_df

    def _currency_convert_df(self, values_df: pd.DataFrame, currency: str) -> pd.DataFrame:
        """Assumes that values_df is a DataFrame with tickers as column headers and dates as index, start_date and
        end_date is either of type date or datetime and currency is a string. First, the method finds the correct FX
        data based on the currency that each ticker is quoted in. The method then converts the values in the DataFrame.
        """
        price_currency = capital_letter_no_blanks(currency)
        logger.info('Converts DataFrame to {}.'.format(price_currency))
        ticker_currency_dict = self.get_ticker_underlying_attribute_dict(list(values_df), Underlying.currency)
        ticker_fx_ticker_dict = {ticker: price_currency + '_' + base_currency + '.FX' if base_currency != price_currency else None for ticker, base_currency in ticker_currency_dict.items()}
        unique_fx_ticker_list = list(set(ticker_fx_ticker_dict.values()))
        try:
            # if None is not in the list python raises a ValueError -> list.remove(x): x not in list
            unique_fx_ticker_list.remove(None)
        except ValueError:
            pass
        if len(unique_fx_ticker_list) == 0:
            return values_df
        logger.debug('Download FX data.')
        fx_total_df = self.get_close_price_df(unique_fx_ticker_list, start_date=values_df.index[0], end_date=values_df.index[-1])
        fx_quote_for_each_ticker_df = select_rows_from_dataframe_based_on_sub_calendar(fx_total_df, values_df.index)
        logger.debug('Create the FX DataFrame.')
        fx_quote_for_each_ticker_df = pd.DataFrame\
            (
                {
                    ticker: fx_quote_for_each_ticker_df.loc[:, ticker_fx_ticker_dict[ticker]]
                    if ticker_fx_ticker_dict[ticker] is not None
                    else 1  # in case the base currency is the same as the price currency
                    for ticker in list(values_df)
                }
            )
        return values_df.mul(fx_quote_for_each_ticker_df)

    def get_open_price_df(self, tickers: {str, list}, start_date: {date, datetime}=None, end_date: {date, datetime}=None,
                          currency: str = None, ffill_na: bool = False, bfill_na: bool = False, drop_na: bool = False)->pd.DataFrame:
        return self._get_open_high_low_close_volume_dividend_df(table=OpenPrice, tickers=tickers, start_date=start_date,
                                                                end_date=end_date, currency=currency, ffill_na=ffill_na,
                                                                bfill_na=bfill_na, drop_na=drop_na)

    def get_high_price_df(self, tickers: {str, list}, start_date: {date, datetime}=None, end_date: {date, datetime}=None,
                          currency: str = None, ffill_na: bool = False, bfill_na: bool = False, drop_na: bool = False)->pd.DataFrame:
        return self._get_open_high_low_close_volume_dividend_df(table=HighPrice, tickers=tickers, start_date=start_date,
                                                                end_date=end_date, currency=currency, ffill_na=ffill_na,
                                                                bfill_na=bfill_na, drop_na=drop_na)

    def get_low_price_df(self, tickers: {str, list}, start_date: {date, datetime}=None, end_date: {date, datetime}=None,
                          currency: str = None, ffill_na: bool = False, bfill_na: bool = False, drop_na: bool = False)->pd.DataFrame:
        return self._get_open_high_low_close_volume_dividend_df(table=LowPrice, tickers=tickers, start_date=start_date,
                                                                end_date=end_date, currency=currency, ffill_na=ffill_na,
                                                                bfill_na=bfill_na, drop_na=drop_na)

    def get_close_price_df(self, tickers: {str, list}, start_date: {date, datetime}=None, end_date: {date, datetime}=None,
                          currency: str = None, ffill_na: bool = False, bfill_na: bool = False, drop_na: bool = False)->pd.DataFrame:
        return self._get_open_high_low_close_volume_dividend_df(table=ClosePrice, tickers=tickers, start_date=start_date,
                                                                end_date=end_date, currency=currency, ffill_na=ffill_na,
                                                                bfill_na=bfill_na, drop_na=drop_na)

    def get_volume_df(self, tickers: {str, list}, start_date: {date, datetime}=None, end_date: {date, datetime}=None,
                      ffill_na: bool = False, bfill_na: bool = False, drop_na: bool = False)->pd.DataFrame:
        return self._get_open_high_low_close_volume_dividend_df(table=Volume, tickers=tickers, start_date=start_date,
                                                                end_date=end_date, currency=None, ffill_na=ffill_na,
                                                                bfill_na=bfill_na, drop_na=drop_na)

    def get_liquidity_df(self, tickers: {str, list}, start_date: {date, datetime}=None, end_date: {date, datetime}=None,
                         currency: str = None, ffill_na: bool = False, bfill_na: bool = False, drop_na: bool = False):
        logger.info('Get liquidity data' + logger_time_interval_message(start_date, end_date))
        close_price_df = self.get_close_price_df(tickers=tickers, start_date=start_date, end_date=end_date, currency=currency,
                                                 ffill_na=ffill_na, bfill_na=bfill_na, drop_na=drop_na)
        volume_df = self.get_volume_df(tickers=tickers, start_date=start_date, end_date=end_date, ffill_na=ffill_na,
                                       bfill_na=bfill_na, drop_na=drop_na)
        volume_df = select_rows_from_dataframe_based_on_sub_calendar(volume_df, close_price_df.index)
        liquidity_df = close_price_df.multiply(volume_df)
        return liquidity_df

    def get_dividend_df(self, tickers: {str, list}, start_date: {date, datetime}=None, end_date: {date, datetime}=None,
                        currency: str = None)->pd.DataFrame:
        return self._get_open_high_low_close_volume_dividend_df(table=Dividend, tickers=tickers, start_date=start_date,
                                                                end_date=end_date, currency=currency, ffill_na=False,
                                                                bfill_na=False, drop_na=False)

    def get_total_return_close_price_df(self, tickers: {str, list}, start_date: datetime = None, end_date: datetime = None,
                                        withholding_tax: {float} = 0.0, currency: str = None, ffill_na: bool = False,
                                        bfill_na: bool = False, drop_na: bool = False):
        logger.info('Get total return data' + logger_time_interval_message(start_date, end_date))
        dividends = self.get_dividend_df(tickers=tickers, start_date=start_date, end_date=end_date)
        close_price_df = self.get_close_price_df(tickers=tickers, start_date=start_date, end_date=end_date, currency=currency,
                                                 ffill_na=ffill_na, bfill_na=bfill_na, drop_na=drop_na)
        if dividends.empty:
            logger.info('No dividends paid' + logger_time_interval_message(start_date, end_date)
                        + 'Returning close price.')
            return close_price_df
        else:
            close_price_local_ccy = close_price_df.copy()
            close_price_roll_if_na = close_price_local_ccy.fillna(method='ffill')
            dividend_yield = dividends.divide(close_price_roll_if_na.shift()) * (1.0 - withholding_tax)
            dividend_yield = dividend_yield.loc[close_price_local_ccy.index]  # same index as the price DataFrame
            close_price_return = close_price_roll_if_na.pct_change()
            total_return = close_price_return + dividend_yield.fillna(value=0)
            cum_total_return = (1.0 + total_return.fillna(value=0)).cumprod()
            index_first_non_nan = close_price_roll_if_na.notna().idxmax()  # index of first non-NaN for each column
            first_value = np.diag(close_price_local_ccy.loc[index_first_non_nan])  # get the first non-NaN for each column
            cum_total_return *= first_value  # to have the same initial value as the original DataFrame
            cum_total_return += close_price_local_ccy * 0.0  # have NaN at same places as original DataFrame
            # convert the total return series into another currency if applicable.
            if isinstance(currency, str):
                return self._currency_convert_df(cum_total_return, currency)
            else:
                return cum_total_return

    # ------------------------------------------------------------------------------------------------------------------
    # get set functionality and static methods
    @staticmethod
    def reformat_tickers(ticker: {str, list}, convert_to_list=False, sort=False) -> {str, list}:
        """Assumes that ticker is either a string or a list and convert_to_list is bool. Returns a string or a list
        of strings where all the strings have capital letters and blanks have been replaced with '_'."""
        # ticker = capital_letter_no_blanks(ticker)
        if isinstance(ticker, list):
            adj_tickers = [tick.upper() for tick in ticker]
            ticker = adj_tickers
        else:
            ticker = ticker.upper()
        if isinstance(ticker, list) and sort:
            ticker.sort()
        if isinstance(ticker, str) and convert_to_list:
            ticker = [ticker]
        return ticker

    # session is read-only
    @property
    def session(self):
        return self._session

    @property
    def database_name(self):
        return self._database_name

    @property
    def database_echo(self):
        return self._database_echo

    # when either database_name or database_echo changes, the session attribute resets using the _set_session method
    @database_name.setter
    def database_name(self, database_name: str):
        self._database = database_name
        self._set_session()

    @database_echo.setter
    def database_echo(self, database_echo: bool):
        self._database_echo = database_echo
        self._set_session()

    def _set_session(self) -> None:
        engine = create_engine(self.database_name, self.database_echo)
        Base.metadata.create_all(engine)  # Create the database tables (these are empty at this stage)
        Session = sessionmaker(bind=engine)  # ORM's 'handle' to the database (bound to the engine object)
        self._session = Session()

    def reset_database(self) -> None:
        """Resets the database, meaning that contents in all tables will be deleted."""
        logger.info('Resetting database ({}).'.format(self.database_name))
        Base.metadata.drop_all(self.session.get_bind())
        Base.metadata.create_all(self.session.get_bind())
        return

    def __repr__(self):
        return f"<FinancialDatabase(name = {self.database_name})>"


class _DataFeeder(FinancialDatabase):
    """Class definition for _DataFeeder. Adds functionality to create and update data."""

    def __init__(self, database_name: str, database_echo=False):
        super().__init__(database_name, database_echo)
        self._data_table_list = [OpenPrice, HighPrice, LowPrice, ClosePrice, Volume, Dividend]

    def add_underlying(self, tickers: {str, list}, refresh_data_after_adding_underlying: bool = True) -> None:
        """Assumes that ticker is either a string or a list of strings. For each ticker the script downloads the
        information of the underlying and its data and inserts it to the database."""
        tickers = self.reformat_tickers(tickers, convert_to_list=True)
        original_tickers = tickers.copy()
        logger.info("Attempts to populate the database with {} ticker(s).\nTicker: {}".format(len(original_tickers),
                                                                                              original_tickers))
        # Remove the tickers that already exists
        tickers = [ticker for ticker in tickers if not self.underlying_exist(ticker)]
        if len(tickers) == 0:
            logger.info('All {} ticker(s) already exist in the database.'.format(len(original_tickers)))
            return
        elif len(tickers) < len(original_tickers):  # if some tickers already existed in the database
            tickers_that_already_exist = list(set(original_tickers).difference(tickers))
            tickers_that_already_exist.sort()
            logger.info('{} ticker(s) already exists.\nTicker: {}'.format(len(tickers_that_already_exist),
                                                                          tickers_that_already_exist))
        logger.info("Populate the database with {} new ticker(s).\nTicker: {}".format(len(tickers), tickers))

        self._populate_underlying_table(tickers)
        if refresh_data_after_adding_underlying:
            self.refresh_data_for_tickers(tickers, delete_existing_data=False)
        return

    def refresh_data_for_tickers(self, tickers: {str, list}, delete_existing_data: bool = True) -> None:
        """Assumes that tickers is either a string or list of strings. Refreshes the OHLC, Volume and dividend data up
        to today."""
        tickers = self.reformat_tickers(tickers, convert_to_list=True)
        for ticker in tickers:  # check for existence. Only add or refresh data if ticker already exists
            if not self.underlying_exist(ticker):
                raise ValueError("{} does not exist in the database.\nUse 'add_underlying(<ticker>) to add it to the "
                                 "database'")
        tickers = self._control_tickers_before_refresh(tickers)  # some tickers might be de-listed...
        if len(tickers) == 0:  # no tickers to refresh or add data to
            return
        start_date, end_date = self._get_start_end_dates_before_refresh(tickers)
        self._refresh_dividends(tickers, start_date, end_date, delete_existing_data)  # refresh dividends
        self._refresh_open_high_low_close_volume(tickers, start_date, end_date, delete_existing_data)  # refresh OHLC and Volume
        self._update_obs_date_and_dividend_history_status(tickers)  # update the dates
        return

    def _get_start_end_dates_before_refresh(self, ticker_list: list)-> tuple:
        """
        Returns the start and end date for the refresh of OHLC, volume and dividends.
        :param ticker_list: list of strings with tickers
        :return: Returns a tuple containing start and end date
        """
        ticker_latest_obs_date_dict = self.get_ticker_underlying_attribute_dict(ticker_list, Underlying.latest_observation_date_with_values)
        if datetime.today().weekday() < 5:  # 0-6 represent the consecutive days of the week, starting from Monday.
            end_date = date.today()  # weekday
        else:
            end_date = date.today() - BDay(1)  # previous business day
        if None in ticker_latest_obs_date_dict.values():  # True if a ticker has no data: Load entire history!
            start_date = None
        else:
            start_date = min(ticker_latest_obs_date_dict.values())

        return start_date, end_date

    def _control_tickers_before_refresh(self, tickers: list, number_of_nan_days_threshold: int = 14) -> list:
        """Assumes tickers is a list of strings and number_of_nan_days_threshold is an int. Returns a new list of tickers
        (strings) where each ticker has no more than number_of_nan_days_threshold days between last observation date and
        last observation date with values. Also if the last observation date WITH VALUES equals today, remove the ticker
        since there is no need to refresh."""
        last_obs_date_dict = self.get_ticker_underlying_attribute_dict(tickers, Underlying.latest_observation_date)
        last_obs_date_with_values_dict = self.get_ticker_underlying_attribute_dict(tickers,
                                                                                   Underlying.latest_observation_date_with_values)
        if datetime.today().weekday() < 5:  # 0-6 represent the consecutive days of the week, starting from Monday.
            end_date = date.today()  # weekday
        else:
            end_date = date.today() - BDay(1)  # previous business day
        for ticker in tickers.copy():
            if last_obs_date_with_values_dict[ticker] is not None:
                num_nan_days = (last_obs_date_with_values_dict[ticker] - last_obs_date_dict[ticker]).days
                if last_obs_date_with_values_dict[ticker] == end_date:
                    tickers.remove(ticker)
                    logger.info("{} is up-to-date.".format(ticker))
                elif num_nan_days > number_of_nan_days_threshold:
                    tickers.remove(ticker)
                    logger.warning("{} has not published a value for the past {} days.".format(ticker, num_nan_days))
        if len(tickers) == 0:
            logger.info("All tickers are either up-to-date or not active.")
        return tickers

    def _populate_underlying_table(self, ticker_list: list):
        """Populates the Underlying table with rows based on the provided list of tickers. This method will be
        overridden in sub classes depending on the API used (e.g. Yahoo finance or Bloomberg)."""
        raise TypeError('_populate_underlying_table should not be called by an instance of a '
                        '_DynamicFinancialDatabase object')

    def _refresh_dividends(self, ticker_list: list, start_date: {date, datetime}=None,
                           end_date: {date, datetime}=None, delete_existing_data: bool = True) -> None:
        """Populate the dividend table with ex-dividend dates and dividend amounts."""
        logger.debug('Refresh dividends for {} ticker(s)'.format(len(ticker_list))
                     + logger_time_interval_message(start_date, end_date))

        # remove the tickers of non dividend paying assets
        excluded_underlying_types = ['INDEX', 'FUTURE']
        query_eligible_tickers = []
        for ticker_sub_list in list_grouper(ticker_list, 500):
            sub_query_eligible_tickers = self.session.query(
                Underlying.ticker
            ).filter(
                and_(
                    Underlying.ticker.in_(ticker_sub_list),
                    ~Underlying.underlying_type.in_(excluded_underlying_types)
                )
            ).all()
            query_eligible_tickers.extend(sub_query_eligible_tickers)

        ticker_list = [tup[0] for tup in query_eligible_tickers]  # list of the eligible tickers
        if len(ticker_list) == 0:
            logger.debug('All underlyings are of type %s.' % ' or '.join(excluded_underlying_types))
            return

        dividend_df = self._retrieve_dividend_df(ticker_list, start_date, end_date)
        if dividend_df is None or dividend_df.empty:
            return
        unique_dates_eligible_to_deletion = list(set(list(dividend_df['ex_div_date'].values)))
        if delete_existing_data:
            self._delete_open_high_low_close_volume_dividend_data(table=Dividend, ticker_list=ticker_list,
                                                                  date_list=unique_dates_eligible_to_deletion)
        logger.debug('Append rows to the Dividend table in the database.')
        dividend_df.to_sql(Dividend.__tablename__, self._session.get_bind(), if_exists='append', index=False)
        logger.debug('Commit the new Dividend rows.')
        self.session.commit()
        return

    def _retrieve_dividend_df(self, ticker_list: list, start_date: {date, datetime}, end_date: {date, datetime}):
        raise TypeError('_get_dividend_df should not be called by an instance of a _DynamicFinancialDatabase '
                        'object')

    def _refresh_open_high_low_close_volume(self, ticker_list: list, start_date: {date, datetime}=None,
                                            end_date: {date, datetime}=None, delete_existing_data: bool = True) -> None:
        """Populate the OpenPrice, HighPrice, LowPrice, ClosePrice and Volume tables with new rows."""
        logger.debug('Refresh OHLC and volume for {} ticker(s)'.format(len(ticker_list))
                     + logger_time_interval_message(start_date, end_date))
        open_high_low_close_volume_df = self._retrieve_open_high_low_close_volume_df(ticker_list, start_date, end_date)
        if open_high_low_close_volume_df is None or open_high_low_close_volume_df.empty:
            return
        data_table_ex_div_list = self._data_table_list.copy()
        data_table_ex_div_list.remove(Dividend)

        for data_table in data_table_ex_div_list:
            logger.debug("Append rows to the {} table in the database.".format(data_table.__tablename__))
            value_df = open_high_low_close_volume_df[open_high_low_close_volume_df['data_type'] == data_table.__valuename__].copy()
            unique_dates_eligible_to_deletion = list(set(list(value_df['obs_date'].values)))
            if delete_existing_data:
                self._delete_open_high_low_close_volume_dividend_data(table=data_table, ticker_list=ticker_list,
                                                                      date_list=unique_dates_eligible_to_deletion)
            value_df.drop('data_type', axis=1, inplace=True)
            value_df.rename(columns={'value': data_table.__valuename__}, inplace=True)
            value_df.to_sql(data_table.__tablename__, self.session.get_bind(), if_exists='append', index=False)
        logger.debug('Commit the new OHLC and Volume rows.')
        self.session.commit()
        return

    def _retrieve_open_high_low_close_volume_df(self, ticker_list: list, start_date: {date, datetime},
                                                end_date: {date, datetime}) -> pd.DataFrame:
        """Should return a DataFrame with 'data_type', 'obs_date', 'value', 'comment', 'data_source', 'underlying_id' as
        column headers"""
        raise TypeError('_get_open_high_low_close_volume_df should not be called with an instance of a '
                        '_DynamicFinancialDatabase object')

    def _ticker_adjustment(self, ticker_list: list):
        """

        :param ticker_list:
        :return:
        """
        adjusted_ticker_list = [self._convert_fx_ticker(ticker) if ticker.endswith('.FX') else ticker for ticker in ticker_list]
        return adjusted_ticker_list

    @staticmethod
    def _convert_fx_ticker(original_ticker)->str:
        """
        Return a string that has replaced the .FX suffix with another format.
        :param original_ticker: str
        :return: str
        """
        return original_ticker

    def __repr__(self):
        return f"<_DynamicFinancialDatabase(name = {self.database_name})>"


class YahooFinanceFeeder(_DataFeeder):
    """Class definition of YahooFinanceFeeder.
    Using the Yahoo Finance API, this class can add and create data to the database."""

    def _populate_underlying_table(self, ticker_list: list) -> None:
        yf_ticker_list = self.yahoo_finance_ticker(ticker_list)
        underlying_list = []
        counter = 0
        for yf_ticker in yf_ticker_list:
            try:
                progression_bar(counter + 1, len(yf_ticker_list))
                logger.debug('Fetching data dictionary from Yahoo Finance for {}...'.format(yf_ticker.ticker))
                ticker_info = yf_ticker.info  # retrieves a dictionary with data e.g. name and sector (takes a while)
            except KeyError:
                raise ValueError("'{}' does not exist as a ticker on Yahoo Finance.".format(yf_ticker.ticker))
            default_str = 'NA'  # in case the attribute does not exist, use a default string
            # e.g. it is normal for an INDEX or FX rate to not have a website
            underlying = Underlying(ticker=ticker_list[counter],
                                    underlying_type=capital_letter_no_blanks(ticker_info.get('quoteType', default_str)),
                                    long_name=capital_letter_no_blanks(ticker_info.get('longName')),
                                    short_name=capital_letter_no_blanks(ticker_info.get('shortName', default_str)),
                                    sector=capital_letter_no_blanks(ticker_info.get('sector', default_str)),
                                    industry=capital_letter_no_blanks(ticker_info.get('industry', default_str)),
                                    currency=capital_letter_no_blanks(ticker_info.get('currency', default_str)),
                                    country=capital_letter_no_blanks(ticker_info.get('country', default_str)),
                                    city=capital_letter_no_blanks(ticker_info.get('city', default_str)),
                                    address=ticker_info.get('address1', default_str),
                                    description=ticker_info.get('longBusinessSummary', default_str),
                                    website=ticker_info.get('website', default_str),
                                    exchange=capital_letter_no_blanks(ticker_info.get('exchange', default_str)))
            underlying_list.append(underlying)
            counter += 1
        logger.info('Append {} row(s) to the Underlying table in the database.'.format(len(underlying_list)))
        self.session.add_all(underlying_list)
        logger.debug('Commit the new Underlying rows.')
        self.session.commit()
        return

    def _retrieve_dividend_df(self, ticker_list: list, start_date: {date, datetime}, end_date: {date, datetime}) \
            -> {pd.DataFrame, None}:
        logger.info("Downloading dividend data from Yahoo Finance and reformat the DataFrame.")
        yf_ticker_list = self.yahoo_finance_ticker(ticker_list)  # need to download the dividends per YF ticker
        dividend_amount_total_df = None  # initialize the resulting DataFrame.
        ticker_underlying_id_dict = self.get_ticker_underlying_attribute_dict(ticker_list, Underlying.id)
        counter = 0
        for yf_ticker in yf_ticker_list:  # loop through each Yahoo Finance Ticker object
            if start_date is None:
                if end_date is None:
                    yf_historical_data_df = yf_ticker.history(period='max')  # the maximum available data
                else:
                    yf_historical_data_df = yf_ticker.history(period='max', end=end_date + timedelta(days=1))
            else:
                if end_date is None:
                    yf_historical_data_df = yf_ticker.history(period='max', start=start_date)
                else:
                    yf_historical_data_df = yf_ticker.history(start=start_date, end=end_date + timedelta(days=1))
            if yf_historical_data_df.empty:
                continue
            yf_historical_data_df = yf_historical_data_df[start_date:]  # handle case when start_date is a holiday
            # yf_historical_data_df contains Open, High, Low, Close, Volume, Dividends and Stock Splits
            dividend_df = yf_historical_data_df.reset_index()[['Date', 'Dividends']]  # extract dates and dividend
            dividend_amount_df = dividend_df[dividend_df['Dividends'] != 0]  # remove the rows with zero. Now the dates
            # are the ex-dividend dates
            dividend_amount_df.insert(0, 'underlying_id', ticker_underlying_id_dict[ticker_list[counter]])
            if dividend_amount_total_df is None:
                dividend_amount_total_df = dividend_amount_df.copy()
            else:
                # for each ticker, combine the DataFrames
                dividend_amount_total_df = pd.concat([dividend_amount_total_df, dividend_amount_df], ignore_index=True)
            counter += 1
        if dividend_amount_total_df is None or dividend_amount_total_df.empty:
            return
        # add comment, name of data source and rename and reshuffle the columns
        dividend_amount_total_df['comment'] = 'Loaded at {}'.format(str(datetime.today())[:19])
        dividend_amount_total_df['data_source'] = 'YAHOO_FINANCE'
        dividend_amount_total_df.rename(columns={'Date': 'ex_div_date', 'Dividends': 'dividend_amount'}, inplace=True)
        return dividend_amount_total_df[['ex_div_date', 'dividend_amount', 'comment', 'data_source', 'underlying_id']]

    def _retrieve_open_high_low_close_volume_df(self, ticker_list: list, start_date: {date, datetime},
                                                end_date: {date, datetime}) -> {pd.DataFrame, None}:
        logger.info("Downloading OHLC and volume data from Yahoo Finance and reformat the DataFrame.")
        multiple_ticker_str = self.multiple_ticker_string(ticker_list)  # ['ABC', 'DEF'] -> 'ABC DEF'
        yf_historical_data_df = yfinance.download(tickers=multiple_ticker_str, start=start_date,
                                                  end=end_date + timedelta(days=1))  # need to add an extra date
        if yf_historical_data_df.empty:
            return
        yf_historical_data_df = yf_historical_data_df.loc[start_date:]
        # E.g. using the two tickers 'ABB.ST' and 'MCD' the DataFrame yf_historical_data_df has the below shape:
        #              Adj Close              ...     Volume
        #                 ABB.ST         MCD  ...     ABB.ST        MCD
        # Date                                ...
        # 1966-07-05         NaN    0.002988  ...        NaN   388800.0
        # 1966-07-06         NaN    0.003148  ...        NaN   687200.0
        # 1966-07-07         NaN    0.003034  ...        NaN  1853600.0
        yf_historical_data_unstacked_df = yf_historical_data_df.unstack().reset_index()
        if len(ticker_list) == 1:  # add the ticker if there is only one ticker (gets removed by default)
            yf_historical_data_unstacked_df.insert(loc=1, column='ticker', value=ticker_list[0])
        yf_historical_data_unstacked_df.columns = ['data_type', 'ticker', 'obs_date', 'value']
        # convert back the tickers from Yahoo Finance format
        # first create a dictionary with tickers with and without the Yahoo Finance format
        ticker_yahoo_finance_format_dict = dict(zip(self._ticker_adjustment(ticker_list), ticker_list))
        # remove the tickers that does not need replacement (when they are the same i.e. key = value)
        ticker_yahoo_finance_format_dict = {key: value for key, value in ticker_yahoo_finance_format_dict.items()
                                            if key != value}
        yf_historical_data_unstacked_df = yf_historical_data_unstacked_df.replace({'ticker': ticker_yahoo_finance_format_dict})
        # add comment and name of data source
        yf_historical_data_unstacked_df['comment'] = 'Loaded at {}'.format(str(datetime.today())[:19])
        yf_historical_data_unstacked_df['data_source'] = 'YAHOO_FINANCE'

        # replace the tickers with the corresponding underlying id
        ticker_underlying_id_dict = self.get_ticker_underlying_attribute_dict(ticker_list, Underlying.id)
        yf_historical_data_unstacked_df['underlying_id'] = \
            yf_historical_data_unstacked_df['ticker'].apply(lambda ticker: ticker_underlying_id_dict[ticker])
        yf_historical_data_unstacked_df.drop(['ticker'], axis=1, inplace=True)

        # clean up by removing NaN and zeros
        yf_historical_data_unstacked_clean_df = yf_historical_data_unstacked_df[
            pd.notnull(yf_historical_data_unstacked_df['value'])].copy()  # remove rows with NaN in data column
        # yf_historical_data_unstacked_clean_df = yf_historical_data_unstacked_clean_df[
        #     yf_historical_data_unstacked_clean_df['value'] != 0].copy()  # remove rows with 0 in data column
        yf_historical_data_unstacked_clean_df['data_type'] = yf_historical_data_unstacked_clean_df['data_type'].str.lower()
        yf_historical_data_unstacked_clean_df = yf_historical_data_unstacked_clean_df[yf_historical_data_unstacked_clean_df['data_type'].isin(['open', 'high', 'low', 'close', 'volume'])].copy()
        yf_historical_data_unstacked_clean_df['data_type'] = yf_historical_data_unstacked_clean_df['data_type'].astype(str) + '_quote'
        return yf_historical_data_unstacked_clean_df[['data_type', 'obs_date', 'value', 'comment', 'data_source',
                                                      'underlying_id']]

    def yahoo_finance_ticker(self, tickers: list):
        adjusted_ticker_list = self._ticker_adjustment(tickers)
        # create a list of Yahoo Finance ticker objects
        yf_tickers = [yfinance.Ticker(ticker) for ticker in adjusted_ticker_list]
        return yf_tickers

    def multiple_ticker_string(self, ticker_list: list) -> str:
        """Assumes that ticker_list is a list of strings. Method returns a string containing each ticker as a
        substring. E.g. the list ['TICKER_A', 'TICKER_B'] yields 'TICKER_A TICKER_B'"""
        adjusted_ticker_list = self._ticker_adjustment(ticker_list)
        result_string = '%s' % ' '.join(adjusted_ticker_list)
        return result_string

    @staticmethod
    def _convert_fx_ticker(original_ticker) -> str:
        """
        Return a string that has replaced the .FX suffix with another format.
        :param original_ticker: str
        :return: str
        """
        original_ticker = original_ticker.replace('.FX', '')  # remove the suffix
        if original_ticker.endswith('_USD'):  # in Yahoo Finance, if the base currency is USD, the 'USD' is omitted
            original_ticker = original_ticker.replace('_USD', '')
        else:
            original_ticker = original_ticker.split('_')[1] + original_ticker.split('_')[0]
        original_ticker += '=X'  # add the Yahoo Finance FX suffix
        return original_ticker

    def __repr__(self):
        return f"<YahooFinancialDatabase(name = {self.database_name})>"


class ExcelFeeder(_DataFeeder):
    """Class definition of ExcelFeeder.
    Loads data from an excel file and stores it in the database"""

    def __init__(self, database_name: str, full_path: str = None, database_echo=False):
        super().__init__(database_name, database_echo)
        self.full_path = full_path
        self._eligible_sheet_names = [Underlying.__tablename__]
        self._eligible_sheet_names.extend([data_table.__tablename__ for data_table in self._data_table_list])
        self._eligible_col_names_underlying = ["ticker", "underlying_type", "long_name", "short_name", "sector", "industry", "country", "city", "address", "currency", "description", "website", "exchange"]
        self._dataframe_sheet_name_dict = {data_table.__tablename__: None for data_table in self._data_table_list}
        self._dataframe_sheet_name_dict.update({Underlying.__tablename__: None})
        self._delete_dates_between_start_end = False

    @staticmethod
    def chose_excel_file_get_file_name()-> str:
        Tk().withdraw()
        file = filedialog.askopenfile(initialdir=__DATABASE_FEED_EXCEL_FILES_FOLDER__,
                                      title='Select excel file containing the underlying.')
        if file is None:
            raise ValueError('file selection failed')
        else:
            return file.name

    def load_data_from_excel(self):
        if self.full_path is None:
            self.full_path = self.chose_excel_file_get_file_name()
        sheet_names = list(self._dataframe_sheet_name_dict.keys())
        for sheet_name in sheet_names:  # loop through the eligible sheet names and save the DataFrames
            try:
                first_column_index = False if sheet_name == 'underlying' else True
                loaded_df = load_df(full_path=self.full_path, sheet_name=sheet_name, first_column_index=first_column_index,
                                    sheet_name_error_handling=False)
                loaded_df = self._clean_loaded_dataframe(loaded_df, sheet_name)
                self._dataframe_sheet_name_dict[sheet_name] = loaded_df
            except xlrd.biffh.XLRDError:
                logger.info("Sheet '{}' did not exist.".format(sheet_name))
                self._dataframe_sheet_name_dict[sheet_name] = None
            except IndexError:
                logger.info("Sheet '{}' is empty.".format(sheet_name))
                self._dataframe_sheet_name_dict[sheet_name] = None

    def _clean_loaded_dataframe(self, loaded_df: pd.DataFrame, data_name: str)-> pd.DataFrame:
        """
        Performs a check on the DataFrame (format etc). The check can be different depending on the name.
        :param loaded_df:
        :param data_name:
        :return:
        """
        default_str = 'NA'
        if len(list(loaded_df.index)) == 0:
            logger.info("Sheet '{}' is empty.".format(data_name))
            loaded_df = None
        else:
            if data_name == 'underlying':
                loaded_df.replace(np.nan, default_str, regex=True, inplace=True)  # in case the attribute does not exist, use a default string
                loaded_df = loaded_df[self._eligible_col_names_underlying].copy()  # only select the eligible column names

                # capital letters and replace blanks with '_'
                col_to_be_capitalized = ['ticker', 'underlying_type', 'sector', 'industry', 'currency', 'country', 'city']
                loaded_df[col_to_be_capitalized] = loaded_df[col_to_be_capitalized].replace(to_replace=' ', value='_', regex=True)
                loaded_df[col_to_be_capitalized] = loaded_df[col_to_be_capitalized].apply(lambda col: col.str.upper())
            elif data_name in [data_table.__tablename__ for data_table in self._data_table_list]:
                # make sure the index is an ascending DatetimeIndex
                if type(loaded_df.index) != pd.DatetimeIndex:
                    raise ValueError("index is not a DatetimeIndex for sheet '{}'".format(data_name))
                elif not loaded_df.index.is_monotonic_increasing:
                    raise ValueError("index is not monotonic increasing for sheet '{}'".format(data_name))
                loaded_df.columns = self.reformat_tickers(list(loaded_df))  # reformat the column names (i.e. tickers)
            else:
                raise ValueError("Data name '{}' not recognized.".format(data_name))
            logger.info("Sheet '{}' loaded successfully.".format(data_name))
        return loaded_df

    def insert_data_to_database(self):
        # populate the underlying table in the database using the specific DataFrame
        if self._dataframe_sheet_name_dict['underlying'] is not None:
            ticker_list = list(self._dataframe_sheet_name_dict['underlying']['ticker'].values)
            self.add_underlying(ticker_list, refresh_data_after_adding_underlying=False)
        else:
            logger.info('No underlying to add from excel file.')

        # get all the unique tickers that existed in each data frame
        tickers = []  # initalizing the list of tickers
        for sheet_name in [data_table.__tablename__ for data_table in self._data_table_list]:
            df = self._dataframe_sheet_name_dict[sheet_name]
            if df is not None:
                tickers.extend(list(df))
        tickers = self.reformat_tickers(tickers, convert_to_list=True)  # capital letters and _ instead of blanks
        tickers = list(set(tickers))  # only unique tickers

        for ticker in tickers:  # check for existence. Only add or refresh data if ticker already exists
            if not self.underlying_exist(ticker):
                raise ValueError("{} does not exist in the database.\nUse 'add_underlying(<ticker>) to add it to the "
                                 "database'".format(ticker))
        self._refresh_dividends(tickers, None, None)  # refresh dividends
        self._refresh_open_high_low_close_volume(tickers, None, None)  # refresh OHLC and Volume
        self._update_obs_date_and_dividend_history_status(tickers)  # update the dates

    def refresh_data_for_tickers(self, tickers: {str, list}) -> None:
        raise NotImplementedError("Can't refresh a given ticker if using ExcelFeeder object")

    def _populate_underlying_table(self, ticker_list: list) -> None:
        total_underlying_df = self._dataframe_sheet_name_dict['underlying']
        underlying_df = total_underlying_df[total_underlying_df['ticker'].isin(ticker_list)].copy()
        underlying_list = []
        for ticker in ticker_list:
            underlying = Underlying(ticker=ticker,
                                    underlying_type=capital_letter_no_blanks(underlying_df['underlying_type'].values[0]),
                                    long_name=capital_letter_no_blanks(underlying_df['long_name'].values[0]),
                                    short_name=capital_letter_no_blanks(underlying_df['short_name'].values[0]),
                                    sector=capital_letter_no_blanks(underlying_df['sector'].values[0]),
                                    industry=capital_letter_no_blanks(underlying_df['industry'].values[0]),
                                    currency=capital_letter_no_blanks(underlying_df['currency'].values[0]),
                                    country=capital_letter_no_blanks(underlying_df['country'].values[0]),
                                    city=capital_letter_no_blanks(underlying_df['city'].values[0]),
                                    address=underlying_df['address'].values[0],
                                    description=underlying_df['description'].values[0],
                                    website=underlying_df['website'].values[0],
                                    exchange=capital_letter_no_blanks(underlying_df['exchange'].values[0]))
            underlying_list.append(underlying)
        logger.info('Append {} row(s) to the Underlying table in the database.'.format(len(underlying_list)))
        self.session.add_all(underlying_list)
        logger.debug('Commit the new Underlying rows.')
        self.session.commit()

    def _retrieve_dividend_df(self, ticker_list, start_date, end_date) -> pd.DataFrame:
        logger.debug("Reformatting dividend DataFrame from Excel.")
        dividend_df = self._dataframe_sheet_name_dict[Dividend.__tablename__]
        if dividend_df is None:
            return
        dividend_df = dividend_df.unstack().reset_index()
        if len(list(dividend_df)) != 3:
            raise ValueError("The dividend DataFrame loaded from excel is not in the correct format.")
        dividend_df.columns = ['ticker', 'ex_div_date', 'dividend_amount']  # rename columns
        dividend_df = dividend_df[dividend_df['ticker'].isin(ticker_list)].copy()  # only the eligible tickers
        dividend_df = dividend_df[dividend_df['dividend_amount'] != 0].copy()  # non-zero dividend amounts only
        dividend_df['comment'] = 'Loaded at {}'.format(str(datetime.today())[:19])
        dividend_df['data_source'] = 'EXCEL'
        ticker_underlying_id_dict = self.get_ticker_underlying_attribute_dict(ticker_list, Underlying.id)
        dividend_df['underlying_id'] = dividend_df['ticker'].apply(lambda ticker: ticker_underlying_id_dict[ticker])
        return dividend_df[['ex_div_date', 'dividend_amount', 'comment', 'data_source', 'underlying_id']]

    def _retrieve_open_high_low_close_volume_df(self, ticker_list, start_date, end_date) -> pd.DataFrame:
        logger.debug("Reformatting OHLC and volume DataFrames from Excel.")
        value_df = None
        comment = 'Loaded at {}'.format(str(datetime.today())[:19])
        for data_table in self._data_table_list:
            value_sub_df = self._dataframe_sheet_name_dict[data_table.__tablename__]
            if value_sub_df is None:
                continue
            value_sub_df = value_sub_df.unstack().reset_index()
            value_sub_df.columns = ['ticker', 'obs_date', 'value']
            value_sub_df['data_type'] = data_table.__valuename__
            if value_df is None:
                value_df = value_sub_df.copy()
            else:
                # for each table, combine the DataFrames
                value_df = pd.concat([value_df, value_sub_df], ignore_index=True)
        # find the corresponding underlying id for each ticker
        ticker_underlying_id_dict = self.get_ticker_underlying_attribute_dict(ticker_list, Underlying.id)
        value_df['underlying_id'] = value_df['ticker'].apply(lambda ticker: ticker_underlying_id_dict[ticker])
        value_df['comment'] = comment
        value_df['data_source'] = 'EXCEL'
        return value_df[['data_type', 'obs_date', 'value', 'comment', 'data_source', 'underlying_id']]

    def __repr__(self):
        return f"<ExcelFeeder(name = {self.database_name})>"


class BloombergFeeder(_DataFeeder):

    def __init__(self, database_name: str, database_echo: bool=False, bbg_echo: bool=True):
        super().__init__(database_name, database_echo)
        self.bbg_con = BloombergConnection(bbg_echo)

    def _populate_underlying_table(self, ticker_list: list)->None:
        fields = ['SECURITY_TYP2', 'SECURITY_NAME', 'SHORT_NAME', 'GICS_SECTOR_NAME', 'GICS_INDUSTRY_NAME',
                  'COUNTRY_FULL_NAME', 'CITY_OF_DOMICILE', 'CRNCY', 'CIE_DES', 'COMPANY_WEB_ADDRESS', 'EXCH_CODE']
        bbg_ticker_list = self._ticker_adjustment(ticker_list)
        underlying_info = self.bbg_con.get_underlying_information(bbg_ticker_list, fields)
        default_str = 'N/A'  # in case attribute does not exist
        underlying_info.replace(to_replace='nan', value=default_str, inplace=True)
        underlying_list = []
        counter = 0
        for ticker in bbg_ticker_list:
            progression_bar(counter + 1, underlying_info.shape[0])
            underlying = Underlying(ticker=ticker_list[counter],
                                    underlying_type=capital_letter_no_blanks(underlying_info.loc[ticker, 'SECURITY_TYP2']),
                                    long_name=capital_letter_no_blanks(underlying_info.loc[ticker, 'SECURITY_NAME']),
                                    short_name=capital_letter_no_blanks(underlying_info.loc[ticker, 'SHORT_NAME']),
                                    sector=capital_letter_no_blanks(underlying_info.loc[ticker, 'GICS_SECTOR_NAME']),
                                    industry=capital_letter_no_blanks(underlying_info.loc[ticker, 'GICS_INDUSTRY_NAME']),
                                    country=capital_letter_no_blanks(underlying_info.loc[ticker, 'COUNTRY_FULL_NAME']),
                                    city=capital_letter_no_blanks(underlying_info.loc[ticker, 'CITY_OF_DOMICILE']),
                                    address=default_str,
                                    currency=underlying_info.loc[ticker, 'CRNCY'].upper().replace(' ', '_'),
                                    description=underlying_info.loc[ticker, 'CIE_DES'],
                                    website=underlying_info.loc[ticker, 'COMPANY_WEB_ADDRESS'],
                                    exchange=capital_letter_no_blanks(underlying_info.loc[ticker, 'EXCH_CODE']))
            underlying_list.append(underlying)
            counter += 1
        logger.info('Append {} row(s) to the Underlying table in the database.'.format(len(underlying_list)))
        self.session.add_all(underlying_list)
        self._add_expiry_date_in_description(ticker_list)  # in case there are tickers that are future contracts
        logger.debug('Commit the new Underlying rows.')
        self.session.commit()

    def _add_expiry_date_in_description(self, tickers: list):
        """
        For the given tickers, check if the underlying type is 'FUTURE'. If True, then adjust the descriptuon to include
        the expiry date of the futures contract: 'EXPIRY date'
        :param tickers: list
        :return: None
        """
        # for all the tickers that are futures, adjust the description to include the expiry date
        # make query to the database
        query_future_tickers = []
        for ticker_sub_list in list_grouper(tickers, 500):
            sub_query_future_tickers = self.session.query(
                Underlying.ticker
            ).filter(
                and_(
                    Underlying.underlying_type == 'FUTURE',
                    Underlying.ticker.in_(ticker_sub_list)
                )
            ).all()
            query_future_tickers.extend(sub_query_future_tickers)

        # list of the tickers that are futures
        future_tickers = [tup[0] for tup in query_future_tickers]

        if len(future_tickers):
            # get the expiry date for all the futures from Bloomberg
            ex_dates_bbg = self.bbg_con.get_underlying_information(future_tickers, 'LAST_TRADEABLE_DT')

            # get the underlying id for all the futures from the database
            ticker_underlying_id_dict = self.get_ticker_underlying_attribute_dict(future_tickers, Underlying.id)

            # loop through all the futures contracts and update the description
            for future in future_tickers:
                desc = 'EXPIRY ' + ex_dates_bbg.loc[future].values[0]
                self.session.query(
                    Underlying
                ).filter(
                    Underlying.id == ticker_underlying_id_dict[future]
                ).update(
                    {'description': desc}
                )

    def _retrieve_dividend_df(self, ticker_list: list, start_date: {date, datetime}=None, end_date: {date, datetime}=None):
        logger.info('Downloading dividend data from Bloomberg and reformat the DataFrame.')
        ticker_underlying_id_dict = self.get_ticker_underlying_attribute_dict(ticker_list, Underlying.id)
        bbg_ticker_list = self._ticker_adjustment(ticker_list)
        dividend_amount_df = self.bbg_con.get_dividend_data(bbg_ticker_list, start_date, end_date, do_pivot=False)
        if dividend_amount_df is None:
            return
        dividend_amount_df['comment'] = 'Loaded at {}.'.format(str(date.today()))
        dividend_amount_df['data_source'] = 'BLOOMBERG'
        # first create a dictionary with tickers with and without the adjusting the FX tickers
        ticker_bloomberg_format_dict = dict(zip(self._ticker_adjustment(ticker_list), ticker_list))
        # remove the tickers that does not need replacement (when they are the same i.e. key = value)
        ticker_bloomberg_format_dict = {key: value for key, value in ticker_bloomberg_format_dict.items() if key != value}
        dividend_amount_df = dividend_amount_df.replace({'ticker': ticker_bloomberg_format_dict})
        dividend_amount_df['underlying_id'] = dividend_amount_df['ticker'].map(ticker_underlying_id_dict)
        dividend_amount_df.rename(columns={'ex_date': 'ex_div_date'}, inplace=True)
        dividend_amount_df['ex_div_date'] = pd.to_datetime(dividend_amount_df['ex_div_date'])
        return dividend_amount_df[['ex_div_date', 'dividend_amount', 'comment', 'data_source', 'underlying_id']]

    def _retrieve_open_high_low_close_volume_df(self, ticker_list: list, start_date: {date, datetime}, end_date: {date, datetime}):
        logger.info('Download OHLC and volume data from Bloomberg and reformat the DataFrame.')
        fields = ['PX_OPEN', 'PX_HIGH', 'PX_LOW', 'PX_LAST', 'PX_VOLUME']
        bbg_ticker_list = self._ticker_adjustment(ticker_list)
        ohlc_volume_bbd_df = self.bbg_con.get_daily_data(bbg_ticker_list, fields, start_date, end_date)
        if ohlc_volume_bbd_df.empty:
            return
        ohlc_volume_bbd_df = ohlc_volume_bbd_df.stack().reset_index().melt(['date', 'field'])  # 'reverse' pivot
        ohlc_volume_bbd_df.rename(columns={'date': 'obs_date', 'field': 'data_type'}, inplace=True)  # same names as database

        # add a column with the underlying id and remove the ticker column
        ticker_underlying_id_dict = self.get_ticker_underlying_attribute_dict(ticker_list, Underlying.id)
        # first create a dictionary with tickers with and without the adjusting the FX tickers
        ticker_bloomberg_format_dict = dict(zip(self._ticker_adjustment(ticker_list), ticker_list))
        # remove the tickers that does not need replacement (when they are the same i.e. key = value)
        ticker_bloomberg_format_dict = {key: value for key, value in ticker_bloomberg_format_dict.items() if key != value}
        ohlc_volume_bbd_df = ohlc_volume_bbd_df.replace({'ticker': ticker_bloomberg_format_dict})
        ohlc_volume_bbd_df['underlying_id'] = ohlc_volume_bbd_df['ticker'].map(ticker_underlying_id_dict)
        ohlc_volume_bbd_df.drop('ticker', inplace=True, axis=1)  # remove the ticker column

        # clean up by removing nan and add more information columns
        ohlc_volume_bbd_clean_df = ohlc_volume_bbd_df[pd.notnull(ohlc_volume_bbd_df['value'])].copy()  # remove NaN
        field_name_dict = {'PX_OPEN': 'open_quote', 'PX_HIGH': 'high_quote', 'PX_LOW': 'low_quote', 'PX_LAST': 'close_quote',
                           'PX_VOLUME': 'volume_quote'}
        ohlc_volume_bbd_clean_df['data_type'] = ohlc_volume_bbd_clean_df['data_type'].map(field_name_dict)
        ohlc_volume_bbd_clean_df['comment'] = 'Loaded at {}.'.format(str(date.today()))
        ohlc_volume_bbd_clean_df['data_source'] = 'BLOOMBERG'
        ohlc_volume_bbd_clean_df['obs_date'] = pd.to_datetime(ohlc_volume_bbd_clean_df['obs_date'])
        return ohlc_volume_bbd_clean_df[['data_type', 'obs_date', 'value', 'comment', 'data_source', 'underlying_id']]

    def reformat_tickers(self, ticker: {str, list}, convert_to_list=False, sort=False):
        ticker = super().reformat_tickers(ticker, convert_to_list, sort)
        ticker = self.bbg_con.add_bbg_ticker_suffix(ticker)
        return ticker

    @staticmethod
    def _convert_fx_ticker(original_ticker) -> str:
        """
        Return a string that has replaced the .FX suffix with another format.
        :param original_ticker: str
        :return: str
        """
        original_ticker = original_ticker.replace('.FX', '')  # remove the suffix
        original_ticker = original_ticker.split('_')[1] + original_ticker.split('_')[0] + ' BGN CURNCY'
        return original_ticker

    def __repr__(self):
        return f"<BloombergFeeder(name = {self.database_name})>"


class DataGetter:
    """Class definition of DataGetter"""

    def __init__(self, tickers: list = None, observation_calendar: pd.DatetimeIndex = None, data_observation_window: int=None,
                 data_observation_frequency: {str, int}=None, data_type: str = 'close', currency: str = None,
                 clean_data: bool = True, drop_na: bool = False, handle_na: str = 'ffill'):
        """
        Used to load data from the database and convert it in a non standard format (list of numpy arrays for each
        date in the observation calendar). Also includes methods that takes price data and calculates metrics such as
        realized volatility
        :param tickers: list containing str or a list of sub-lists containing str
        :param observation_calendar: pd.DatetimeIndex
        :param data_observation_window: int or str (name of weekday)
        :param data_observation_frequency: int
        :param data_type: str
        :param currency: str
        :param clean_data: bool if true, nan are rolled forward
        :param drop_na: bool if true, for each column, nan is dropped such that the above rows with non nan is shifted
        down
        """
        if clean_data and drop_na:
            logger.warning('since clean_data is True, setting drop_na to True has no effect since all nan are rolled '
                           'forward')
        self._underlying_data = None
        self._prev_start_date = None  # used to check if data needs to be downloaded
        self._prev_end_date = None
        self._weekday_i_dict = {'mon': 0, 'monday': 0, 'tuesday': 1, 'tue': 1, 'wed': 2, 'wednesday': 2, 'thursday': 3,
                                'thu': 4, 'friday': 5, 'fri': 5}
        self._available_data_types = ['open', 'high', 'low', 'close', 'total_return', 'volume', 'liquidity']
        self._fin_db = FinancialDatabase(__MY_DATABASE_NAME__)

        self.tickers = tickers
        self.observation_calendar = observation_calendar
        self.observation_window = data_observation_window
        self.data_observation_frequency = data_observation_frequency  # TODO only str? (int does not make sense anymore...)
        self._data_type = data_type
        self._currency = currency
        self.clean_data = clean_data
        self.drop_na = drop_na
        self.handle_na = handle_na
        self._handle_na_methods = [None, 'skip', 'drop', 'ffill']  # TODO comment

    def get_underlying_df(self, return_lag: int = None):
        """
        Return a DataFrame with the underlying data adjusted for frequency, dropping or forward filling nan
        :param return_lag: int
        :return: DataFrame
        """
        # download data if necessary
        if self._data_needs_updating():
            self._download_underlying_data()

        # retrieve the data for the relevant tickers
        underlying_df = self._underlying_data[self.tickers].copy()

        # adjust the observation frequency
        underlying_df = self._adjust_underlying_data_frequency(underlying_df=underlying_df)

        # clean the data when applicable
        if self.handle_na == 'drop':
            underlying_df.dropna(inplace=True)
        elif self.handle_na == 'ffill':
            underlying_df.fillna(method='ffill')

        # format change
        if return_lag:
            if return_lag >= 1:
                nan_or_1 = underlying_df[~underlying_df.isnull()] = 1  # set all non NaN to 1
                underlying_df = underlying_df.fillna(method='ffill').pct_change(return_lag, fill_method=None)
                underlying_df *= nan_or_1
            else:
                raise ValueError('return_lag needs to be an int larger or equal to 1')
        return underlying_df

    def get_holding_period_return(self, ending_lag: int = 0, ffill_values: bool = False):
        """
        Calculate the rolling holding period return for each column
        Holding period return for stock = Price(t - ending_lag) / Price(t - observation_window) - 1
        :param ending_lag: int
        :param ffill_values: bool if True, replace nan with the previous available non-nan financial time series value
        :return: DataFrame
        """
        self._price_data_method_logger_msg()
        underlying_df = self.get_underlying_df()
        performance_df = fin_ts.holding_period_return(multivariate_df=underlying_df, lag=self.observation_window,
                                                      ending_lag=ending_lag, skip_nan=self.handle_na == 'skip')
        return self._adjust_financial_time_series_result(multivariate_df=performance_df, ffill_values=ffill_values)

    def get_volatility(self, return_lag: int = 1, ffill_values: bool = False):
        """
        Returns a DataFrame with realized volatility for each observation date
        :param return_lag: int
        :param ffill_values: bool if True, replace nan with the previous available non-nan financial time series value
        :return: DataFrame
        """
        self._price_data_method_logger_msg()
        underlying_df = self.get_underlying_df()
        vol_df = fin_ts.realized_volatility(multivariate_df=underlying_df, return_lag=return_lag, skip_nan=self.handle_na == 'skip',
                                            window=self.observation_window, annualization_factor=self.get_annualization_factor())
        return self._adjust_financial_time_series_result(multivariate_df=vol_df, ffill_values=ffill_values)

    def get_beta(self, beta_instrument_name: str,  return_lag: int = 1, ffill_values: bool = False):
        """
        Returns a DataFrame with beta with respect to the given instrument for each observation date
        :param beta_instrument_name: str name of the column to calculate beta against
        :param return_lag: int
        :param ffill_values: bool if True, replace nan with the previous available non-nan financial time series value
        :return: DataFrame
        """
        self._price_data_method_logger_msg()
        underlying_df = self.get_underlying_df()
        beta_df = fin_ts.realized_beta(multivariate_df=underlying_df, beta_instrument_name=beta_instrument_name,
                                       return_lag=return_lag, skip_nan=self.handle_na == 'skip', window=self.observation_window)
        return self._adjust_financial_time_series_result(multivariate_df=beta_df, ffill_values=ffill_values)

    def _adjust_financial_time_series_result(self, multivariate_df: pd.DataFrame, ffill_values: bool):
        """
        Replace nan with the previous non-nan data if applicable and lookup the data for the specific observation
        calendar
        :param multivariate_df: DataFrame
        :param ffill_values: bool
        :return: DataFrame
        """
        if ffill_values:
            # roll forward the last non-nan values
            multivariate_df.fillna(method='ffill', inplace=True)

        # observe the financial time series data on the observation dates (lookup the latest available value in case the
        # observation date does not exists
        original_calendar = multivariate_df.index  # calendar for the financial time series data
        adj_obs_date_index_list = [original_calendar.get_loc(obs_date, method='ffill')
                                   for obs_date in self.observation_calendar if obs_date >= min(original_calendar)]
        # lookup the data
        data_for_adj_obs_date = multivariate_df.to_numpy()[adj_obs_date_index_list, :]  # more efficient(?) to do lookup with numpy array

        # in case the observation dates are before the first available date in the database, add nans to the first rows
        if len(adj_obs_date_index_list) != len(self.observation_calendar):  # not enough dates in the financial database
            # add nan for observation dates where no data exists in the financial database
            num_missing_rows = len(self.observation_calendar) - len(adj_obs_date_index_list)
            nan_array = np.empty((num_missing_rows, data_for_adj_obs_date.shape[1]))
            nan_array[:] = np.NaN
            # 'stack' the data on top of rows with nan
            data_for_adj_obs_date = np.vstack([nan_array, data_for_adj_obs_date])
        return pd.DataFrame(data=data_for_adj_obs_date, index=self.observation_calendar,
                            columns=multivariate_df.columns)

    def get_annualization_factor(self) -> float:
        """
        Calculate an annualization factor as the ratio of 252 and the observation frequency
        :return: float
        """
        if isinstance(self.data_observation_frequency, str):
            annualization_factor = 252 / 5
        elif self.data_observation_frequency is None:
            annualization_factor = 252
        else:
            annualization_factor = 252 / self.data_observation_frequency
        return annualization_factor

    def _adjust_underlying_data_frequency(self, underlying_df: pd.DataFrame):
        # observe weekly data based on the data_observation_frequency str variable
        if isinstance(self.data_observation_frequency, str):
            underlying_df = underlying_df[self._underlying_data.index.weekday
                                            == self._weekday_i_dict[self.data_observation_frequency]]
        elif isinstance(self.data_observation_frequency, int):
            # sort index in descending order. this is done to have the count start from the latest observation date
            underlying_df = underlying_df.sort_index(ascending=False).iloc[::self.data_observation_frequency, :].sort_index()
        else:
            pass
        return underlying_df

    def _download_underlying_data(self):
        """
        Downloads the data based on the specific attributes and stores it
        :return: None
        """
        self._check_parameters()
        ffill = self.clean_data
        start_date = self._get_start_date()
        end_date = self._get_end_date()

        # load the data from the database on the specific data type e.g. 'close'
        if self.data_type == self._available_data_types[0]:
            self._underlying_data = self._fin_db.get_open_price_df(tickers=self.tickers,
                                                                   start_date=start_date,
                                                                   end_date=end_date,
                                                                   currency=self.currency,
                                                                   ffill_na=ffill)
        elif self.data_type == self._available_data_types[1]:
            self._underlying_data = self._fin_db.get_high_price_df(tickers=self.tickers,
                                                                   start_date=start_date,
                                                                   end_date=end_date,
                                                                   currency=self.currency,
                                                                   ffill_na=ffill)
        elif self.data_type == self._available_data_types[2]:
            self._underlying_data = self._fin_db.get_low_price_df(tickers=self.tickers,
                                                                  start_date=start_date,
                                                                  end_date=end_date,
                                                                  currency=self.currency,
                                                                  ffill_na=ffill)
        elif self.data_type == self._available_data_types[3]:
            self._underlying_data = self._fin_db.get_close_price_df(tickers=self.tickers,
                                                                    start_date=start_date,
                                                                    end_date=end_date,
                                                                    currency=self.currency,
                                                                    ffill_na=ffill)
        elif self.data_type == self._available_data_types[4]:
            self._underlying_data = self._fin_db.get_total_return_close_price_df(tickers=self.tickers,
                                                                                 start_date=start_date,
                                                                                 end_date=end_date,
                                                                                 currency=self.currency,
                                                                                 ffill_na=ffill)
        elif self.data_type == self._available_data_types[5]:
            self._underlying_data = self._fin_db.get_volume_df(tickers=self.tickers,
                                                               start_date=start_date,
                                                               end_date=end_date,
                                                               ffill_na=ffill)

        elif self.data_type == self._available_data_types[6]:
            self._underlying_data = self._fin_db.get_liquidity_df(tickers=self.tickers,
                                                                  start_date=start_date,
                                                                  end_date=end_date,
                                                                  currency=self.currency,
                                                                  ffill_na=ffill)
        else:
            raise ValueError("data type '{}' not recognized".format(self.data_type.lower()))

        # save the start and end dates
        if start_date is None:
            self._prev_start_date = self._underlying_data.index[-1]  # set to the oldest date when data is available
        else:
            self._prev_start_date = start_date
        self._prev_end_date = end_date

        if end_date > self._underlying_data.index[-1]:
            logger.warning("there is no '{}' data between {} and {}".format(self.data_type,
                                                                            str(self._underlying_data.index[-1])[:10],
                                                                            str(end_date)[:10]))

    def _data_needs_updating(self) -> bool:
        """
        Returns True if one of the following is true: 1) first time loading data 2) tickers are missing 3) dates are
        missing, else False
        :return: bool
        """
        if self._underlying_data is None:
            # first time loading data
            return True
        elif len(set(list(self._underlying_data)).difference(self.tickers)) == 0 and self._underlying_data.shape[1] != len(self.tickers):
            # some tickers are not available in the old underlying data
            return True
        elif self._dates_not_available():
            return True
        else:
            return False

    def _check_parameters(self):
        """
        Raises an error in case something is incorrect with the setup before getting the data
        :return: None
        """
        if self.tickers is None:
            raise ValueError('tickers have not been specified')
        elif self.observation_calendar is None:
            raise ValueError('observation_calendar has not been specified')
        elif isinstance(self.tickers[0], list) and len(self.tickers) != len(self.observation_calendar):
            raise ValueError('number of sub-lists with tickers ({}) needs to be the same as the number of observation dates ({})'.format(len(self.tickers), len(self.observation_calendar)))
        else:
            return

    def _dates_not_available(self):
        """
        Returns True is the dates are not up to date
        :return: bool
        """
        start_date = self._get_start_date()
        if self._prev_start_date is None:  # last time data was loaded without a start date
            return self._get_end_date() > self._prev_end_date
        elif start_date is None:
            return True
        else:
            return start_date < self._prev_start_date

    def _get_start_date(self):
        """
        Returns the start date taking into account a buffer
        :return: datetime, None
        """
        if self.observation_window is None:
            return None
        else:
            start_date = min(self.observation_calendar)
            day_buffer = 15  # add a three week buffer
            biz_day_shift = self.observation_window + day_buffer
            if isinstance(self.data_observation_frequency, str):
                biz_day_shift *= 5  # assuming weekly frequency
            elif isinstance(self.data_observation_frequency, int):
                biz_day_shift *= self.data_observation_frequency
            return start_date - BDay(int(biz_day_shift))

    def _get_end_date(self):
        """
        Returns the end i.e. the last available date in the observation calendar
        :return: datetime
        """
        return max(self.observation_calendar)

    def _price_data_method_logger_msg(self):
        if self.data_type not in self._available_data_types[:5]:
            logger.warning("this method is normally used using price data such as: '%s'" % ", ".join(self._available_data_types[:5]))

    # ------------------------------------------------------------------------------------------------------------------
    # get and setter methods
    @property
    def tickers(self):
        return self._tickers

    @tickers.setter
    def tickers(self, tickers: list):
        # ticker can either be 1) a list of str 2) a list of sub-lists with str
        if tickers is None:
            self._tickers = tickers
        else:
            try:
                self._tickers = [e.upper() for e in tickers]
            except AttributeError:
                raise ValueError('tickers must be specified as a list or as a list of sub lists with strings')

    @property
    def data_observation_frequency(self):
        return self._data_observation_frequency

    @data_observation_frequency.setter
    def data_observation_frequency(self, data_observation_frequency: {int, str}):
        if isinstance(data_observation_frequency, str) and data_observation_frequency.lower() in self._weekday_i_dict.keys():
            self._data_observation_frequency = data_observation_frequency
        elif data_observation_frequency is None or data_observation_frequency > 0:
            self._data_observation_frequency = data_observation_frequency
        else:
            raise ValueError("data_observation_frequency needs to be an int larger than 0 or a str equal to '%s'" % "' or '".join(self._weekday_i_dict.keys()))

    @property
    def observation_window(self):
        return self._observation_window

    @observation_window.setter
    def observation_window(self, observation_window: {int, None}):
        if observation_window is None or observation_window > 0:
            self._observation_window = observation_window
        else:
            raise ValueError('observation_window needs to be an int strictly greater than 0 or None')

    @property
    def data_type(self):
        return self._data_type

    @data_type.setter
    def data_type(self, data_type: str):
        if data_type.lower() in self._available_data_types:
            if data_type.lower() != self._data_type:
                # reset the underlying DataFrame if the data type has changed
                self._underlying_data = None
            self._data_type = data_type.lower()
        else:
            raise ValueError("data_type needs to be equal to '%s'" % "' or '".join(self._available_data_types))

    @property
    def currency(self):
        return self._currency

    @currency.setter
    def currency(self, currency: str):
        if currency != self._currency:
            # reset the underlying DataFrame if the currency has changed
            self._underlying_data = None
        self._currency = currency


def logger_time_interval_message(start_date: {date, datetime}, end_date: {date, datetime}) -> str:
    logger_message = ''
    if start_date is not None:
        logger_message += ' from {}'.format(str(start_date)[:10])
    if end_date is not None:
        logger_message += ' up to {}'.format(str(end_date)[:10])
    logger_message += '.'
    return logger_message


def add_data_from_excel_main():
    excel_db = ExcelFeeder(__MY_DATABASE_NAME__)
    excel_db.load_data_from_excel()
    excel_db.insert_data_to_database()


if __name__ == '__main__':
    add_data_from_excel_main()


