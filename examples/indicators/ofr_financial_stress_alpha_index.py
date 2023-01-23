"""ofr_financial_stress_alpha_index.py
Calculate a financial stress signal using research done by OFR

source: https://www.financialresearch.gov/financial-stress-index/
"""

import pandas as pd

from file_paths_HIDDEN import OFR_DATA_PATH


def get_ofr_data():
    """
    Returns a Series with the OFR Financial Stress Index
    :return: Series
    """
    ofr_df = pd.read_csv(OFR_DATA_PATH, index_col=0)
    ofr_df.index = pd.to_datetime(ofr_df.index)
    return ofr_df['OFR FSI']


def get_ofr_fsi_alpha_signal():
    """
    Returns a Series that looks at the rate of change in financial stress which is an indicator of financial conditions
    becoming more fragile.
    Use the OFR Financial Stress Index to calculate a fragility ratio by looking at the difference between a short term
    rolling average and a lagging average. The signal is normalized by dividing with the rolling standard deviation.
    :return: Series
    """
    lead_avg_w = 15
    lagging_avg_w = 252
    std_w = lagging_avg_w

    ofr_fsi = get_ofr_data()

    # (leading avg. - lagging avg.) / standard deviation
    return (ofr_fsi.rolling(window=lead_avg_w).mean() - ofr_fsi.rolling(window=lagging_avg_w).mean()) / ofr_fsi.rolling(window=std_w).std()


