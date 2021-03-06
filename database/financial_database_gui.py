from tkinter import *
from tkinter.ttk import Combobox
import tkinter.messagebox as msg
from tkinter import filedialog
from tkcalendar import *

import webbrowser

import pandas as pd
from datetime import date
import logging

from database.models_db import Underlying
from database.financial_database import YahooFinanceFeeder, BloombergFeeder, FinancialDatabase
from database.config_database import __MY_DATABASE_NAME__, __EXCEL_TICKER_FOLDER__, __DATA_REQUEST_FOLDER__
from excel_tools import save_df, format_requested_data_workbook
from general_tools import list_grouper, progression_bar

__DEFAULT_FONT__ = ("Arial", 11)
__DEFAULT_BOLD_FONT__ = __DEFAULT_FONT__ + ('bold', )

data_source_list = ['Yahoo Finance', 'Bloomberg', 'Huggorm']
method_list = ['Manually', 'Using Excel', 'Using URL', 'Using attribute filter']
action_method_dict = {'Add underlying': [method_list[0], method_list[1], method_list[2]],
                      'Refresh underlying': [method_list[0], method_list[1], method_list[3]],
                      'Delete underlying': [method_list[0], method_list[1], method_list[3]],
                      'Download data': [method_list[0], method_list[1], method_list[3]]}
action_data_source_dict = {'Add underlying': [data_source_list[0], data_source_list[1]],
                      'Refresh underlying': [data_source_list[0], data_source_list[1]],
                      'Delete underlying': [data_source_list[2]],
                      'Download data': [data_source_list[2]]}

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s : %(module)s : %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class HandleUnderlyingWindow(Tk):
    """Class definition of HandleUnderlyingWindow"""

    def __init__(self):
        super().__init__()
        Label(self, text="Hey!\n\nHere you are able to refresh, add and delete underlying assets in the financial "
                         "database.\n\nWhat would you like to do today?\n",
              justify='l', font=__DEFAULT_FONT__).grid()
        self.ticker_list = None

        # add frame
        frame = Frame(self)  # , bd=1, relief=SUNKEN)
        Label(frame, text='Chose action and method', font=__DEFAULT_BOLD_FONT__, justify='l').grid(columnspan=2)

        # action drop-down list
        Label(frame, text='Action:', justify='l', font=__DEFAULT_FONT__).grid(row=1)
        self.action_combo = Combobox(frame)
        self.action_combo.bind('<<ComboboxSelected>>', self.update_method_and_data_source_combo_box)
        # self.action_combo.bind('<<ComboboxSelected>>', self.update_data_source_combo_box)

        self.action_combo['values'] = tuple(action_method_dict.keys())
        self.action_combo.grid(row=1, column=1)

        # method drop-down list (dependent on the action drop-list)
        Label(frame, text='Method:', justify='l', font=__DEFAULT_FONT__).grid(row=2)
        self.method_combo = Combobox(frame)
        self.method_combo.grid(row=2, column=1)

        # method drop-down list (dependent on the action drop-list)
        Label(frame, text='Data source:', justify='l', font=__DEFAULT_FONT__).grid(row=3)
        self.data_source_combo = Combobox(frame)
        self.data_source_combo.grid(row=3, column=1)

        Button(frame, text='Perform action', fg='green', font=__DEFAULT_FONT__,
               command=self.retrieve_tickers_click).grid(row=4, pady=10, columnspan=2)
        frame.grid(row=1, sticky='w')

    def update_method_and_data_source_combo_box(self, event=None):
        self.method_combo['values'] = action_method_dict[self.action_combo.get()]
        self.method_combo.current(0)
        self.data_source_combo['values'] = action_data_source_dict[self.action_combo.get()]
        self.data_source_combo.current(0)

    def retrieve_tickers_click(self):
        if self.action_combo.get() == '':
            msg.showinfo('Warning!', 'Please select an action')
        elif self.method_combo.get() == '':
            msg.showinfo('Warning!', 'Please select a method')
        else:
            HandleUnderlyingWindow.chose_action = self.action_combo.get()
            title = self.action_combo.get() + ' ' + self.method_combo.get().lower()
            if self.method_combo.get() == method_list[0]:  # manual input
                ManualInputWindow(self, title=title)
            elif self.method_combo.get() == method_list[1]:  # using excel
                ExcelInputWindow(self, title=title)
            elif self.method_combo.get() == method_list[2]:  # using an URL (Yahoo finance)
                # URLInputWindow(self, title=title)
                msg.showinfo('URL information', "Tickers can not be retrieved using URL at the moment.")
            elif self.method_combo.get() == method_list[3]:  # setting up an attribute filter
                AttributeFilterInputWindow(self, title=title)
            else:
                raise ValueError('Method not yet configured')


class _InputWindow(Toplevel):
    """Class definition of _InputWindow"""

    def __init__(self, parent=None, title: str = None):
        Toplevel.__init__(self, parent)
        self.transient(parent)  # associate this window with a parent window
        if title:  # if it is provided, add a title
            self.title(title)
        self.result = None
        self.parent = parent  # parent window
        self.create_widgets()
        self.grab_set()  # makes sure that no mouse or keyboard events are sent to the wrong window
        self.wait_window()  # local event loop

    def apply_result(self):
        if self.result is not None:
            action = self.parent.action_combo.get()

            data_source = self.parent.data_source_combo.get()
            # initialize a database handler used to add and refresh data
            batch_size = None
            if data_source == data_source_list[0]:
                batch_size = 15
                fin_db = YahooFinanceFeeder(__MY_DATABASE_NAME__)
            elif data_source == data_source_list[1]:
                batch_size = 100
                fin_db = BloombergFeeder(__MY_DATABASE_NAME__, bbg_echo=False)
            else:
                fin_db = FinancialDatabase(__MY_DATABASE_NAME__)

            if action == 'Add underlying':
                counter = 1
                list_of_ticker_list = list_grouper(self.result, batch_size)
                tickers_not_available = []
                for ticker_sub_list in list_of_ticker_list:
                    progression_bar(counter, len(list_of_ticker_list))
                    try:
                        fin_db.add_underlying(ticker_sub_list)
                    except ValueError:
                        logger.warning('One ticker is not available in {}.'.format(data_source))
                        for ticker in ticker_sub_list:
                            try:
                                fin_db.add_underlying(ticker)
                            except ValueError:
                                logger.warning('{} does not exist as a ticker on {}.'.format(ticker, data_source))
                                tickers_not_available.append(ticker)
                    counter += 1
                logger.info('Done with adding underlying(s) to database.')
                if len(tickers_not_available) > 0:
                    logger.warning('{} ticker(s) could not be added.\nTicker(s): %s'.format(len(tickers_not_available))
                                   % ', '.join(tickers_not_available))
            elif action == 'Refresh underlying':
                counter = 1
                list_of_ticker_list = list_grouper(self.result, 100)
                for ticker_sub_list in list_of_ticker_list:
                    progression_bar(counter, len(list_of_ticker_list))
                    fin_db.refresh_data_for_tickers(ticker_sub_list)
                    counter += 1
                logger.info('Done with refreshing underlying(s) in the database.')
            elif action == 'Delete underlying':
                msg_before_delete = msg.askquestion('Warning', 'Are you sure you want to delete {} ticker(s) from the '
                                                               'database?'.format(len(self.result)), icon='warning')
                if msg_before_delete == 'no':
                    return
                else:
                    fin_db.delete_underlying(self.result)
                logger.info('Done with deleting underlying(s) from database.')
            elif action == 'Download data':
                DataRetrievalWindow(self, title=action, ticker_list=self.result)
            else:
                msg.showinfo('Action not executable', "The action '{}' has not been implemented yet".format(action))
        else:
            raise ValueError('No tickers selected (self.result is None)')

    def create_widgets(self):
        pass

    def cancel(self, event=None):
        self.apply_result()
        self.parent.focus_set()
        self.destroy()


class ManualInputWindow(_InputWindow):
    """Class definition of ManualInputWindow subclass of _InputWindow"""

    def create_widgets(self):
        self.geometry("{}x{}".format(350, 70))
        Label(self, text="Enter tickers below separated by a comma (,).", font=__DEFAULT_FONT__, justify='l').grid()

        # add a text box where you insert tickers as strings separated by a comma (capital letters and remove blanks)
        self.entry = Entry(self, width=50)
        self.entry.grid(row=1, sticky='w', pady=10)
        self.entry.focus()
        self.entry.bind('<Return>', self.press_enter)

    def press_enter(self, event=None):
        txt_result = self.entry.get()  # get the text from the entry box
        split_tickers = txt_result.split(',')
        tickers = []
        for sub_str in split_tickers:
            adj_str = sub_str.rstrip().lstrip().split(' ')
            if len(adj_str) == 1:
                tickers.append(adj_str[0])
            else:
                new_adj_str = adj_str[0]
                for i in range(1, len(adj_str)):
                    if adj_str[i] != '':
                        new_adj_str += ' {}'.format(adj_str[i])
                tickers.append(new_adj_str)
        tickers = [ticker for ticker in tickers if len(ticker) > 0]  # remove any empty strings (mistakenly put ,,)
        self.result = tickers
        self.cancel()


class ExcelInputWindow(_InputWindow):
    """Class definition of ExcelInputWindow subclass of _InputWindow"""
    def __init__(self, parent=None, title: str = None):
        self.result_df = self.select_file_get_df(__EXCEL_TICKER_FOLDER__)
        self.string_var = StringVar()
        super().__init__(parent, title)

    @staticmethod
    def select_file_get_df(init_dir: str):
        file_name = filedialog.askopenfile(initialdir=init_dir,
                                           title='Select excel file containing tickers.').name
        return pd.read_excel(file_name)

    def get_values_from_column(self):
        tickers = list(self.result_df[self.string_var.get()].values)
        self.result = tickers
        self.cancel()

    def create_widgets(self):
        Label(self, text='Chose a column from below to extract the tickers from:', font=__DEFAULT_FONT__).grid()
        counter = 1
        for col_name in list(self.result_df):
            Radiobutton(self, text=col_name, value=col_name, variable=self.string_var).grid(row=counter)
            counter += 1
        self.string_var.set(list(self.result_df)[0])  # select the first column
        Button(self, text='Get tickers', fg='green', command=self.get_values_from_column).grid(row=counter)


class URLInputWindow(_InputWindow):
    """Class definition of URLInputWindow"""

    def create_widgets(self):
        Label(self, text='On the Yahoo finance website you are able to generate a stock filter based \non certain '
                         'criteria e.g. country and size of market capitalization.', justify=LEFT,
              font=__DEFAULT_FONT__).grid(sticky='W')

        # URL link that is clickable
        yf_url = 'https://finance.yahoo.com/screener/new'
        url_label = Label(self, text=yf_url, justify='l', font=__DEFAULT_FONT__, fg="blue", cursor="hand2")
        url_label.grid(row=1, sticky='w')
        url_label.bind("<Button-1>", lambda e: self.click_url(yf_url))

        # information
        Label(self, text='\nAfter creating the stock filter such that you can see a table with tickers, \ncopy the URL '
                         'and paste it below.', justify='l', font=__DEFAULT_FONT__).grid(row=2, sticky='w')

        # add text box where you paste an URL
        self.url_entry = Entry(self, width=50)
        self.url_entry.grid(row=3, sticky='w', pady=10)
        self.url_entry.focus()
        self.url_entry.bind('<Return>', self.press_enter)

    def press_enter(self, event=None):
        print(self.url_entry.get())
        self.cancel()

    @staticmethod
    def click_url(url: str):
        webbrowser.open_new(url)


class AttributeFilterInputWindow(_InputWindow):
    """Class definition of AttributeFilterInputWindow"""

    def __init__(self, parent=None, title: str = None):
        self.filter_dict = {}
        super().__init__(parent, title)

    def create_widgets(self):
        Label(self, text="Create a filter that is used to select underlyings from the database.", font=__DEFAULT_FONT__,
              justify='l').grid()

        # frame with a drop-down list and an entry box
        frame = Frame(self)
        frame.grid(row=1, sticky='w')

        # drop-down list with attributes
        Label(frame, text='Attribute:', font=__DEFAULT_FONT__, justify='l').grid()
        self.attribute_combo = Combobox(frame)
        attribute_list = Underlying.__table__.columns.keys()
        # remove attributes that we can't use in the current setup
        eligible_attribute_list = list(
            set(attribute_list).difference(
                ('description', 'first_ex_div_date', 'latest_observation_date', 'latest_observation_date_with_values',
                 'oldest_observation_date', 'id', 'ticker', 'has_dividend_history', 'address', 'website')
            )
        )
        eligible_attribute_list.sort()
        self.attribute_combo['values'] = eligible_attribute_list
        self.attribute_combo.grid(row=0, column=1)

        # entry box
        Label(frame, text='Value:', font=__DEFAULT_FONT__, justify='l').grid(row=1)
        self.entry = Entry(frame)
        self.entry.bind('<Return>', self.press_entry)
        self.entry.grid(row=1, column=1)

        # button to run the filter
        Button(frame, text='Run filter', command=self.run_filter_click, fg='green').grid(row=2, columnspan=2)

        # frame that displays the filter
        self.second_frame = Frame(self)
        self.second_frame.grid(row=2)
        Label(self.second_frame, text='\nCurrent filter:', font=__DEFAULT_FONT__).grid()
        self.filter_display_label = Label(self.second_frame, text='None', font=__DEFAULT_FONT__)
        self.filter_display_label.grid(row=1)

    def press_entry(self, event=None):
        if self.attribute_combo.get() == '':
            msg.showinfo('Warnign!', "Please select an attribute")
            # self.master.lift()  # window moves backward for some reason ...
        elif self.entry.get() == '':
            # if the attribute exists in the filter, then it gets removed
            if getattr(Underlying, self.attribute_combo.get()) in list(self.filter_dict.keys()):
                del self.filter_dict[getattr(Underlying, self.attribute_combo.get())]
            pass
        else:
            txt_result = self.entry.get()
            value_list = [s.strip().upper().replace(' ', '_') for s in txt_result.split(',')]
            value_list = [s for s in value_list if len(s)]

            # update the filter
            self.filter_dict.update({getattr(Underlying, self.attribute_combo.get()): value_list})

            # reset the entry box
            self.entry.delete(0, END)
            self.attribute_combo.set('')
        self.display_filter()

    def display_filter(self):
        self.filter_display_label.grid_forget()  # hide the old version of the filter
        filter_text = ''
        for attribute, value_list in self.filter_dict.items():
            filter_text += str(attribute)[len('Underlying.'):] + ' = %s' % ' or '.join(value_list) + '\n'
            if attribute != list(self.filter_dict.keys())[-1]:
                filter_text += 'and\n'
        if filter_text == '':
            filter_text = 'None'
        self.filter_display_label = Label(self.second_frame, text=filter_text, font=__DEFAULT_FONT__)
        self.filter_display_label.grid(row=1)

    def run_filter_click(self):
        if self.filter_dict == {}:  # empty filter
            msg_when_running_without_filter = msg.askquestion('Warning', 'Are you sure you want to ' +
                                                              HandleUnderlyingWindow.chose_action.lower().split()[0] +
                                                              ' all tickers in the database?', icon='warning')
            # self.master.lift()  # window moves backward for some reason ...
            if msg_when_running_without_filter == 'no':
                return
        else:
            pass
        fin_db_handler = FinancialDatabase(__MY_DATABASE_NAME__)
        tickers = fin_db_handler.get_ticker(self.filter_dict)
        self.result = tickers
        self.cancel()


class DataRetrievalWindow(_InputWindow):
    """Class definition of DataRetrievalWindow subclass of _InputWindow"""

    def __init__(self, parent=None, title: str = None, ticker_list: list = None):
        self.ticker_list = ticker_list
        self.available_data_choices = ['Open price', 'High price', 'Low price', 'Close price', 'Volume', 'Liquidity',
                                       'Dividend', 'Data for underlying']
        self.result_df_dict = {}  # list of DataFrames
        self.requested_data_desc = ''
        super().__init__(parent, title)

    def create_widgets(self):
        Label(self, text='Chose what data you want to download.', justify='l', font=__DEFAULT_FONT__).grid()
        self.attribute_combo = Combobox(self)
        self.attribute_combo['values'] = self.available_data_choices
        self.attribute_combo.set(self.available_data_choices[3])
        self.attribute_combo.grid(row=1, sticky='w')
        Button(self, text='Get data', fg='green', command=self.get_data).grid(row=2, sticky='w')
        self.data_display_lable = Label(self, text=self.requested_data_desc, font=__DEFAULT_FONT__)
        self.data_display_lable.grid(row=3)

    def get_data(self):
        data_request = self.attribute_combo.get()
        if data_request == 'Data for underlying':
            UnderlyingDataWindow(self, title=data_request, ticker_list=self.ticker_list)
        elif data_request == 'Open price':
            OpenPriceDataWindow(self, title=data_request, ticker_list=self.ticker_list)
        elif data_request == 'High price':
            HighPriceDataWindow(self, title=data_request, ticker_list=self.ticker_list)
        elif data_request == 'Low price':
            LowPriceDataWindow(self, title=data_request, ticker_list=self.ticker_list)
        elif data_request == 'Close price':
            ClosePriceDataWindow(self, title=data_request, ticker_list=self.ticker_list)
        elif data_request == 'Volume':
            VolumeDataWindow(self, title=data_request, ticker_list=self.ticker_list)
        elif data_request == 'Liquidity':
            LiquidityDataWindow(self, title=data_request, ticker_list=self.ticker_list)
        elif data_request == 'Dividend':
            DividendDataWindow(self, title=data_request, ticker_list=self.ticker_list)
        self.display_data_request()

    def display_data_request(self):
        self.data_display_lable.grid_forget()
        self.requested_data_desc = 'Current data request: '
        for dataframe_name in list(self.result_df_dict.keys()):
            self.requested_data_desc += '\n' + dataframe_name
        self.data_display_lable = Label(self, text=self.requested_data_desc, font=__DEFAULT_FONT__)
        self.data_display_lable.grid(row=3)
        Button(self, text='Save result', fg='green', command=self.save_data_to_excel).grid(row=4)

    def save_data_to_excel(self):
        if self.result_df_dict == {}:
            msg.showinfo('Warning', "There is no data to save.")
            return
        else:
            file_name = filedialog.asksaveasfile(mode='w', initialdir=__DATA_REQUEST_FOLDER__, title='Save requested data.',
                                                 defaultextension='.xlsx').name
            logger.info("Saving data to excel workbook...")
            save_df(list(self.result_df_dict.values()), full_path=file_name, sheet_name_list=list(self.result_df_dict.keys()))
            logger.info("Formatting excel workbook...")
            format_requested_data_workbook(file_name)
            logger.info('Done!')
            self.cancel()

    def cancel(self, event=None):
        self.parent.focus_set()
        self.destroy()


class UnderlyingDataWindow(DataRetrievalWindow):
    """Class definition of UnderlyingDataWindow subclass of DataRetrievalWindow"""
    def __init__(self, parent=None, title: str = None, ticker_list: list = None):
        self.int_vars = []
        attribute_list = Underlying.__table__.columns.keys()
        attribute_list.remove('ticker')
        self.attribute_list = attribute_list
        super().__init__(parent, title, ticker_list)

    def create_widgets(self):
        Label(Label(self, text='Select data', font=__DEFAULT_BOLD_FONT__).grid())
        counter = 1
        for attribute in self.attribute_list:
            var = IntVar()
            Checkbutton(self, text=attribute, variable=var).grid(row=counter, sticky='w')
            self.int_vars.append(var)
            counter += 1
        Button(self, text='Get data', fg='green', command=self.get_underlying_data).grid(row=counter)

    def get_underlying_data(self):
        if any(selected_var.get() == 1 for selected_var in self.int_vars):
            chosen_attributes = [attribute for i, attribute in enumerate(self.attribute_list) if self.int_vars[i].get()]
            fin_db = FinancialDatabase(__MY_DATABASE_NAME__)
            result_df = fin_db.get_underlying_data(self.ticker_list, chosen_attributes)
            self.parent.result_df_dict.update({'underlying_data': result_df})
            logger.info('Done with loading underlying data!')
            self.cancel()
        else:
            msg.showinfo('Warning', "Please select an underlying attribute e.g. 'sector' or 'currency'.")


class DailyDataWindow(DataRetrievalWindow):
    def __init__(self, parent=None, title: str = None, ticker_list: list = None):
        self.start_date = None
        self.end_date = None
        self.currency = None
        super().__init__(parent, title, ticker_list)

    def create_widgets(self):
        Label(self, text='Select dates and format', font=__DEFAULT_BOLD_FONT__).grid(columnspan=2)
        Label(self, text='Start date:', font=__DEFAULT_FONT__).grid(row=1)
        self.start_date_entry = DateEntry(self, width=15, background='green', foreground='white', borderwidth=3,
                                          year=2000, month=1, day=1, date_pattern='mm/dd/y')
        self.start_date_entry.grid(row=1, column=1)
        Label(self, text='End date:', font=__DEFAULT_FONT__).grid(row=2)
        self.end_date_entry = DateEntry(self, width=15, background='green', foreground='white', borderwidth=3,
                                        year=date.today().year, month=date.today().month, day=date.today().day,
                                        date_pattern='mm/dd/y')
        self.end_date_entry.grid(row=2, column=1)
        Label(self, text='Handle missing data:', font=__DEFAULT_FONT__).grid(row=3)
        self.handle_nan_alternatives = ['Do nothing', 'Roll value forward', 'Remove all']
        self.handle_nan_combo = Combobox(self)
        self.handle_nan_combo['values'] = self.handle_nan_alternatives
        self.handle_nan_combo.set(self.handle_nan_alternatives[0])
        self.handle_nan_combo.grid(row=3, column=1)

    def get_start_date(self):
        return self.get_date(self.start_date_entry.get())

    def get_end_date(self):
        return self.get_date(self.end_date_entry.get())

    def handle_nan_df(self, df: pd.DataFrame)-> pd.DataFrame:
        if self.handle_nan_combo.get() == self.handle_nan_alternatives[0]:
            return df
        elif self.handle_nan_combo.get() == self.handle_nan_alternatives[1]:
            return df.fillna(method='ffill')
        elif self.handle_nan_combo.get() == self.handle_nan_alternatives[2]:
            return df.dropna(how='any')

    @staticmethod
    def get_date(_date):
        if len(_date) == 0:
            return None
        else:
            day = int(_date.split('/')[1])
            month = int(_date.split('/')[0])
            year = int(_date.split('/')[-1])
            return date(year, month, day)


class OpenPriceDataWindow(DailyDataWindow):
    def __init__(self, parent=None, title: str = None, ticker_list: list = None):
        super().__init__(parent, title, ticker_list)

    def create_widgets(self):
        super().create_widgets()
        Label(self, text='Currency:', font=__DEFAULT_FONT__).grid(row=4)
        self.currency_entry = Entry(self)
        self.currency_entry.grid(row=4, column=1)
        Button(self, text='Get open price', fg='green', command=self.get_open_price).grid(row=5, columnspan=2)

    def get_currency(self):
        if len(self.currency_entry.get()) == 0:
            return None
        else:
            return self.currency_entry.get()

    def get_open_price(self):
        fin_db = FinancialDatabase(__MY_DATABASE_NAME__)
        price_info = ''
        if self.get_currency() is not None:
            price_info += '_' + self.get_currency().upper()

        result_df = fin_db.get_open_price_df(tickers=self.ticker_list, start_date=self.get_start_date(),
                                             end_date=self.get_end_date(), currency=self.get_currency())
        result_df = self.handle_nan_df(result_df)
        self.parent.result_df_dict.update({'open_price' + price_info: result_df})
        logger.info('Done with loading open price!')
        self.cancel()


class HighPriceDataWindow(DailyDataWindow):
    def __init__(self, parent=None, title: str = None, ticker_list: list = None):
        super().__init__(parent, title, ticker_list)

    def create_widgets(self):
        super().create_widgets()
        Label(self, text='Currency:', font=__DEFAULT_FONT__).grid(row=4)
        self.currency_entry = Entry(self)
        self.currency_entry.grid(row=4, column=1)
        Button(self, text='Get high price', fg='green', command=self.get_high_price).grid(row=5, columnspan=2)

    def get_currency(self):
        if len(self.currency_entry.get()) == 0:
            return None
        else:
            return self.currency_entry.get()

    def get_high_price(self):
        fin_db = FinancialDatabase(__MY_DATABASE_NAME__)
        price_info = ''
        if self.get_currency() is not None:
            price_info += '_' + self.get_currency().upper()

        result_df = fin_db.get_high_price_df(tickers=self.ticker_list, start_date=self.get_start_date(),
                                             end_date=self.get_end_date(), currency=self.get_currency())
        result_df = self.handle_nan_df(result_df)
        self.parent.result_df_dict.update({'high_price' + price_info: result_df})
        logger.info('Done with loading high price!')
        self.cancel()


class LowPriceDataWindow(DailyDataWindow):
    def __init__(self, parent=None, title: str = None, ticker_list: list = None):
        super().__init__(parent, title, ticker_list)

    def create_widgets(self):
        super().create_widgets()
        Label(self, text='Currency:', font=__DEFAULT_FONT__).grid(row=4)
        self.currency_entry = Entry(self)
        self.currency_entry.grid(row=4, column=1)
        Button(self, text='Get low price', fg='green', command=self.get_low_price).grid(row=5, columnspan=2)

    def get_currency(self):
        if len(self.currency_entry.get()) == 0:
            return None
        else:
            return self.currency_entry.get()

    def get_low_price(self):
        fin_db = FinancialDatabase(__MY_DATABASE_NAME__)
        price_info = ''
        if self.get_currency() is not None:
            price_info += '_' + self.get_currency().upper()

        result_df = fin_db.get_low_price_df(tickers=self.ticker_list, start_date=self.get_start_date(),
                                            end_date=self.get_end_date(), currency=self.get_currency())
        result_df = self.handle_nan_df(result_df)
        self.parent.result_df_dict.update({'low_price' + price_info: result_df})
        logger.info('Done with loading low price!')
        self.cancel()


class ClosePriceDataWindow(DailyDataWindow):
    def __init__(self, parent=None, title: str = None, ticker_list: list = None):
        super().__init__(parent, title, ticker_list)

    def create_widgets(self):
        super().create_widgets()
        Label(self, text='Currency:', font=__DEFAULT_FONT__).grid(row=4)
        self.currency_entry = Entry(self)
        self.currency_entry.grid(row=4, column=1)
        Label(self, text='Total return:', font=__DEFAULT_FONT__).grid(row=5)
        self.total_return_combo = Combobox(self)
        self.total_return_combo.bind('<<ComboboxSelected>>', self.update_total_return)
        self.total_return_combo['values'] = ['No', 'Yes']
        self.total_return_combo.set('No')
        self.total_return_combo.grid(row=5, column=1)
        Label(self, text='Tax on dividends (%):', font=__DEFAULT_FONT__).grid(row=6)
        self.div_tax_entry = Entry(self)
        self.div_tax_entry.config(state='disabled')
        self.div_tax_entry.grid(row=6, column=1)
        Button(self, text='Get close price', fg='green', command=self.get_price).grid(row=7, columnspan=2)

    def get_currency(self):
        if len(self.currency_entry.get()) == 0:
            return None
        else:
            return self.currency_entry.get()

    def update_total_return(self, event=None):
        if self.total_return_combo.get() == 'No':
            self.div_tax_entry.config(state='disabled')
        else:
            self.div_tax_entry.config(state='normal')

    def get_div_tax(self):
        if len(self.div_tax_entry.get()) == 0:
            return 0
        else:
            return int(self.div_tax_entry.get()) / 100

    def get_price(self):
        fin_db = FinancialDatabase(__MY_DATABASE_NAME__)
        price_info = ''
        if self.get_currency() is not None:
            price_info += '_' + self.get_currency().upper()
        if self.total_return_combo.get() == 'No':
            result_df = fin_db.get_close_price_df(tickers=self.ticker_list, start_date=self.get_start_date(),
                                                  end_date=self.get_end_date(), currency=self.get_currency())
            result_df = self.handle_nan_df(result_df)
            self.parent.result_df_dict.update({'close_price' + price_info: result_df})
        else:

            result_df = fin_db.get_total_return_df(tickers=self.ticker_list, start_date=self.get_start_date(),
                                                   end_date=self.get_end_date(), currency=self.get_currency(),
                                                   withholding_tax=self.get_div_tax())
            result_df = self.handle_nan_df(result_df)
            self.parent.result_df_dict.update({'total_return_price' + price_info: result_df})
        logger.info('Done with loading price!')
        self.cancel()


class VolumeDataWindow(DailyDataWindow):
    def __init__(self, parent=None, title: str = None, ticker_list: list = None):
        super().__init__(parent, title, ticker_list)

    def create_widgets(self):
        super().create_widgets()
        Button(self, text='Get volume', fg='green', command=self.get_volume).grid(row=4, columnspan=2)

    def get_volume(self):
        fin_db = FinancialDatabase(__MY_DATABASE_NAME__)
        result_df = fin_db.get_volume_df(tickers=self.ticker_list, start_date=self.get_start_date(), end_date=self.get_end_date())
        result_df = self.handle_nan_df(result_df)
        self.parent.result_df_dict.update({'volume': result_df})
        logger.info('Done with loading volume!')
        self.cancel()


class LiquidityDataWindow(DailyDataWindow):
    def __init__(self, parent=None, title: str = None, ticker_list: list = None):
        super().__init__(parent, title, ticker_list)

    def create_widgets(self):
        super().create_widgets()
        Label(self, text='Currency:', font=__DEFAULT_FONT__).grid(row=4)
        self.currency_entry = Entry(self)
        self.currency_entry.grid(row=4, column=1)
        Button(self, text='Get liquidity', fg='green', command=self.get_liquidity).grid(row=5, columnspan=2)

    def get_currency(self):
        if len(self.currency_entry.get()) == 0:
            return None
        else:
            return self.currency_entry.get()

    def get_liquidity(self):
        fin_db = FinancialDatabase(__MY_DATABASE_NAME__)
        price_info = ''
        if self.get_currency() is not None:
            price_info += '_' + self.get_currency().upper()

        result_df = fin_db.get_liquidity_df(tickers=self.ticker_list, start_date=self.get_start_date(),
                                            end_date=self.get_end_date(), currency=self.get_currency())
        result_df = self.handle_nan_df(result_df)
        self.parent.result_df_dict.update({'liquidity' + price_info: result_df})
        logger.info('Done with loading liquidity!')
        self.cancel()


class DividendDataWindow(DailyDataWindow):
    def __init__(self, parent=None, title: str = None, ticker_list: list = None):
        super().__init__(parent, title, ticker_list)

    def create_widgets(self):
        super().create_widgets()
        Label(self, text='Currency:', font=__DEFAULT_FONT__).grid(row=4)
        self.currency_entry = Entry(self)
        self.currency_entry.grid(row=4, column=1)
        Button(self, text='Get dividend', fg='green', command=self.get_dividend).grid(row=5, columnspan=2)

    def get_currency(self):
        if len(self.currency_entry.get()) == 0:
            return None
        else:
            return self.currency_entry.get()

    def get_dividend(self):
        fin_db = FinancialDatabase(__MY_DATABASE_NAME__)
        price_info = ''
        if self.get_currency() is not None:
            price_info += '_' + self.get_currency().upper()

        result_df = fin_db.get_dividend_df(tickers=self.ticker_list, start_date=self.get_start_date(),
                                           end_date=self.get_end_date(), currency=self.get_currency())
        result_df = self.handle_nan_df(result_df)
        self.parent.result_df_dict.update({'dividend' + price_info: result_df})
        logger.info('Done with loading dividend!')
        self.cancel()


def main():
    root = HandleUnderlyingWindow()
    root.title('Handling underlying in the financial database')
    root.mainloop()


if __name__ == '__main__':
    main()

