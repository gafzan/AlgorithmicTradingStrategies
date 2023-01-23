"""filter_handler_old.py"""

import pandas as pd
import numpy as np

from jsonschema import validate as validate_json

from tools.array_tools import sum_nan_arrays


class FilterHandler:
    """Definition of FilterHandler"""

    def __init__(self, filter_collection: (dict, tuple, list) = None):
        """
        Used to implement a series of filters either as a union = ( ) or an intersection = ( )
        :param filter_collection: dict, tuple or list
        """
        self.filter_collection = filter_collection

    def run(self, previous_filter_df: pd.DataFrame = None):
        """
        Runs the filter according to the filter rules in the filter_collection attributes. filter_collection can be a
        tuple of dict, list of dict or a mix of them. A tuple of filter rules are treated as a union of filter rules
        while a list of filters are treated as an intersection of filters
        :param previous_filter_df: DataFrame
        :return:
        """
        return self._run(filter_collection=self.filter_collection, previous_filter_df=previous_filter_df)

    def _run(self, filter_collection: {dict, tuple, list}=None, previous_filter_df: pd.DataFrame = None):

        if isinstance(filter_collection, (tuple, list)):
            # use union operator on the elements in tuple and intersection on elements in list
            is_union = isinstance(filter_collection, tuple)

            result = None if is_union else previous_filter_df
            for filter_rule in filter_collection:
                # for intersection, previous_filter_df is updated after each loop
                # when using a union, the previous_filter_df stays the same
                sub_result = self._run(filter_collection=filter_rule,
                                       previous_filter_df=previous_filter_df if is_union else result)
                if result is None:
                    result = sub_result
                elif is_union:
                    # sum with floor and cap at -1 and 1
                    arr_data = np.clip(sum_nan_arrays(result.values, sub_result.values), -1, 1)
                    # convert to DataFrame
                    result = pd.DataFrame(data=arr_data, columns=result.columns, index=result.index)
                else:
                    # intersect values based on previous filter
                    intersection_a = sub_result.values * result.values
                    # make sure that -1 intersected with -1 is -1 and not 1
                    intersection_a[(sub_result.values == -1) & (result.values == -1)] = -1
                    result = pd.DataFrame(data=intersection_a, columns=result.columns,
                                          index=result.index)
                    # result = sub_result
            return result
        else:
            if isinstance(filter_collection, dict):
                return self._run_filter(filter_rule=filter_collection, previous_filter_df=previous_filter_df)
            else:
                return self._run_filter(filter_rule={'filter': filter_collection, 'position': 'long'},
                                        previous_filter_df=previous_filter_df)

    def _run_filter(self, filter_rule: dict, previous_filter_df: pd.DataFrame = None):
        self.validate_params(**filter_rule)
        filter_obj = filter_rule['filter']
        if previous_filter_df is not None:
            filter_obj.previous_filter_df = previous_filter_df
        position = 1 if filter_rule.get('position', 'long') == 'long' else -1
        return filter_obj.run() * position

    def validate_params(self, **kwargs):
        validate_json(kwargs, schema=self.get_param_schema())

    @staticmethod
    def get_param_schema():
        return {
            'title': 'filter handler',
            'type': 'object',
            'properties': {
                'filter': {
                    'description': 'instance of a filter object',
                },
                'position': {
                    'description': '(optional) long or short the instruments that passes the filter',
                    'type': 'string',
                    'enum': ['long', 'short']
                },
            },
            'required': ['filter']
        }














