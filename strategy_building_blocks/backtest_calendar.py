"""backtest_calendar.py"""

import pandas as pd
import pandas.core.common as com
from pandas.tseries.offsets import BDay
import datetime

FREQUENCY_OPERATOR_MAP = {
    'weekly':
        {
            'TimeStampAttribute': 'week',
            'FrequencyValues': range(1, 53)
        },
    'monthly':
        {
            'TimeStampAttribute': 'month',
            'FrequencyValues': range(1, 13)
        },
    'quarterly':
        {
            'TimeStampAttribute': 'quarter',
            'FrequencyValues': range(1, 5)
        },
    'yearly': {},
    'never': {}
}

WEEKDAY_NUM_MAP = {
    'mon': 0,
    'tue': 1,
    'wed': 2,
    'thu': 3,
    'fri': 4
}


class BacktestCalendar:

    def __init__(self, rebalance_rules: dict = None, reweight_rules: dict = None, review_rules: dict = None,
                 reweight_lag: int = None, review_lag: int = None, rebalance_dates: list = None,
                 reweight_dates: list = None, review_dates: list = None):
        self.rebalance_rules = rebalance_rules
        self.reweight_rules = reweight_rules
        self.review_rules = review_rules
        self.reweight_lag = reweight_lag
        self.review_lag = review_lag
        self.rebalance_dates = rebalance_dates
        self.reweight_dates = reweight_dates
        self.review_dates = review_dates

        # Review date > Reweight date > Rebalance date

        self._rebalance_calendar = None
        self._reweight_calendar = None
        self._review_calendar = None

    def get_calendars(self, as_dataframe: bool = False):
        """
        Either returns a dictionary with the different calendars or a DataFrame with 3 columns
        :param as_dataframe: bool
        :return: dict or DataFrame
        """
        self.generate_calendars()
        cal_dict = {
            'rebalance_calendar': self.rebalance_calendar,
            'reweight_calendar': self.reweight_calendar,
            'review_calendar': self.review_calendar
        }

        if as_dataframe:
            return pd.DataFrame(
                dict(
                    [
                        (name, pd.Series(calendar)) for name, calendar in cal_dict.items()
                    ]
                )
            )
        else:
            return cal_dict

    def generate_calendars(self):
        """
        Generate and set the calendars based on the specified parameters for rebalance, reweight and review calendars
        :return:
        """
        self._check_calendar_param()

        calendar_params = [
            ('rebalance', [self.rebalance_rules, self.rebalance_dates]),
            ('review', [self.review_rules, self.review_dates, self.review_lag]),
            ('reweight', [self.reweight_rules, self.reweight_dates, self.reweight_lag])
            ]

        # loop through each instruction tuple and set the respective calendar attribute
        for cal_param in calendar_params:
            calendar_name = cal_param[0]
            if calendar_name != 'rebalance' and com.count_not_none(*cal_param[1]) == 0:
                calendar = self.rebalance_calendar
            else:
                calendar = self._generate_calendar(params=cal_param[1])
            setattr(self, f"_{calendar_name}_calendar", calendar)  # set calendar attribute

    def _generate_calendar(self, params: list):
        """
        Returns the calendar that is generated by one set of parameters in the given list of sets of parameters based
        on the type
        :param params: list
        :return:
        """
        for param in params:
            if isinstance(param, dict):
                # assumes the parameters are given as a rule dict
                return self._get_calendar_from_rule(cal_rule=param)
            elif isinstance(param, list):
                # assumes the parameters are given as a list of dates
                return self._get_calendar_from_list(date_list=param)
            elif isinstance(param, int):
                # assumes the parameter is given by a avg_lag with respect to the rebalance calendar
                return self._get_calendar_from_lag(lag=param)
        return

    def _check_calendar_param(self):
        if com.count_not_none(self.rebalance_rules, self.rebalance_dates) != 1:
            raise ValueError("Of the two parameters: 'rebalance_rules', 'rebalance_dates' exactly one must be specified")
        if com.count_not_none(self.reweight_rules, self.reweight_dates, self.reweight_lag) > 1:
            raise ValueError("Of the three parameters: 'reweight_rules', 'reweight_dates', 'reweight_lag' not more than "
                             "one must be specified")
        if com.count_not_none(self.review_rules, self.review_dates, self.review_lag) > 1:
            raise ValueError("Of the three parameters: 'review_rules', 'review_dates', 'review_lag' not more than "
                             "one must be specified")

    @staticmethod
    def _get_calendar_from_rule(cal_rule: dict):
        return generate_calendar(**cal_rule)

    @staticmethod
    def _get_calendar_from_list(date_list: list):
        return pd.to_datetime(date_list)

    def _get_calendar_from_lag(self, lag: int):
        return self.rebalance_calendar - BDay(abs(lag))

    # ------------------------------------------------------------------------------------------------------------------
    # the calendars are 'read-only'
    @property
    def rebalance_calendar(self):
        return self._rebalance_calendar

    @property
    def review_calendar(self):
        return self._review_calendar

    @property
    def reweight_calendar(self):
        return self._reweight_calendar


def generate_calendar(start_date: str, end_date: str = None, periods: int = None, frequency: str = None,
                      frequency_values: list = None, days: {int, str, list, tuple}=None, holidays: {list, str}=None,
                      business_day_only: bool = True):
    """
    Generate a calendar in the form of a DatetimeIndex. Allows for specifying a custom frequency e.g. every 2nd Friday
    of every other month: frequency='monthly', days=('fri', 2), frequency_values=[1, 3, 5, 7, 9, 11]
    :param start_date: str
    :param end_date: str
    :param periods: in used in pandas.date_range()
    :param frequency: str if 'weekly', 'monthly', 'quarterly', 'yearly' 'never', it will be a custom calendar, else uses
    pandas.date_range() or pandas.bdate_range()
    :param frequency_values: list
    :param days: int, str, list, tuple for example [5, 'thu', ('fri', 2)] selects the 5th business day, 1st Thursday and
    2nd Friday
    :param holidays: str, list
    :param business_day_only: bool
    :return:
    """
    if frequency in FREQUENCY_OPERATOR_MAP.keys():
        return generate_custom_calendar(frequency=frequency, days=days, start_date=start_date, end_date=end_date,
                                        frequency_values=frequency_values, business_day_only=business_day_only,
                                        holidays=holidays)
    else:
        if business_day_only:
            return pd.bdate_range(start=start_date, end=end_date, periods=periods, freq=frequency, holidays=holidays)
        else:
            return pd.date_range(start=start_date, end=end_date, periods=periods, freq=frequency, holidays=holidays)


def generate_custom_calendar(frequency: str, days: {int, str, list, tuple}, start_date: str, end_date: str = None,
                             frequency_values: list = None, business_day_only: bool = True, holidays: {str, list}=None):
    """
    Generate a DatetimeIndex according to a start date, frequency (e.g. 'monthly'), frequency values (e.g. [1, 3]
    meaning 1st and 3rd month when frequency='monthly'), days and end date.
    :param frequency: str 'weekly', 'monthly', 'quarterly', 'yearly', 'never'
    :param days: int, str, list, tuple
    :param start_date: str
    :param end_date: str (optional)
    :param frequency_values: list
    :param business_day_only: bool (True default)
    :param holidays: str or list of str
    :return:
    """
    frequency = frequency.lower()
    _check_frequency(frequency=frequency, frequency_values=frequency_values)

    # in case end_date is not specified, set it to today
    if end_date is None:
        end_date = datetime.date.today()
    if not isinstance(days, list):
        days = [days]

    if business_day_only:
        # daily business day calendar
        daily_calendar = pd.bdate_range(start=f'1/1/{start_date[-4:]}', end=end_date, holidays=holidays)
    else:
        # daily calendar
        daily_calendar = pd.date_range(start=f'1/1/{start_date[-4:]}', end=end_date, holidays=holidays)

    if frequency == 'never':
        return pd.to_datetime([daily_calendar[0]])
    else:
        # loop through each year and select the days
        custom_dates = []  # initialize the list to store dates
        for year in range(daily_calendar[0].year, daily_calendar[-1].year):
            sub_cal_per_year = daily_calendar[daily_calendar.year == year]  # select the days for the specific year
            for d_select in days:
                if frequency == 'yearly':
                    custom_dates.extend(_date_per_day_selection(calendar=sub_cal_per_year, day_selection=d_select))
                else:
                    if frequency_values is None:
                        frequency_values = FREQUENCY_OPERATOR_MAP[frequency]['FrequencyValues']
                    for freq_val in frequency_values:
                        # create a sub calendar based on the frequency value
                        sub_cal = sub_cal_per_year[getattr(sub_cal_per_year, FREQUENCY_OPERATOR_MAP[frequency]['TimeStampAttribute']) == freq_val]
                        custom_dates.extend(_date_per_day_selection(calendar=sub_cal, day_selection=d_select))
        custom_dates = list(set(custom_dates))
        custom_dates.sort()
        start_date_dt = pd.date_range(start=start_date, periods=1, freq='D')[0]
        return pd.to_datetime([d for d in custom_dates if start_date_dt <= d])  # return all dates after start date (incl.)


def _check_frequency(frequency: str, frequency_values: list):
    """
    Checks so that 'frequency' and 'frequency_values' are specified correctly.
    An error will be raised if elements in 'frequency_values' does not make sense given the frequency.
    E.g. 'frequency' = 'weekly' and 'frequency_values' = [52, 53] is not allowed since there are only 52 weeks in a year.
    :param frequency: str
    :param frequency_values: list
    :return: list
    """
    if frequency not in FREQUENCY_OPERATOR_MAP.keys():
        raise ValueError(f"'{frequency}' is not a recognized frequency. For custom calendars, please choose between "
                         f"'%s'" % "', '".join(FREQUENCY_OPERATOR_MAP.keys()))
    if frequency_values is not None:
        if any((freq_val not in FREQUENCY_OPERATOR_MAP[frequency]['FrequencyValues']) for freq_val in frequency_values):
            raise ValueError("elements in 'frequency_values' are not eligible (e.g. 'frequency' = 'monthly' and "
                             "'frequency_values' = [13] is not allowed)")
    return


def _date_per_day_selection(calendar, day_selection: {int, str, list, tuple}):
    """
    Returns a list of dates selected from 'calendar' according to the rules specified in 'day_selection'
    E.g day_selection = [5, 'thu', ('fri', 2)] selects the 5th business day, 1st Thursday and 2nd Friday
    :param calendar:
    :param day_selection: int, str, list, tuple
    :return:
    """
    ret = []  # initialize the list of dates to return
    if not isinstance(day_selection, list):
        day_selection = [day_selection]

    # loop through all day selection instructions and find the corresponding day and store it in a list
    for d_select in day_selection:
        if isinstance(d_select, int):
            cal_idx = d_select - 1
        elif isinstance(d_select, str):
            cal_idx = 0
            calendar = calendar[calendar.dayofweek == WEEKDAY_NUM_MAP[d_select.lower()[:3]]]
        elif isinstance(d_select, tuple):
            try:
                tpl_int_idx = list(map(type, d_select)).index(int)
                tpl_str_idx = list(map(type, d_select)).index(str)
            except ValueError:
                raise ValueError('day selection tuple needs to contain a str and int')
            if len(d_select) != 2:
                raise ValueError('day selection tuple needs to have length 2')
            cal_idx = d_select[tpl_int_idx] - 1
            day_str = d_select[tpl_str_idx].lower()[:3]
            calendar = calendar[calendar.dayofweek == WEEKDAY_NUM_MAP[day_str]]
        else:
            raise ValueError("element in 'day_selection' can only be int, str, tuple or a list of these types")
        if cal_idx < len(calendar):
            ret.append(calendar[cal_idx])
    return ret
