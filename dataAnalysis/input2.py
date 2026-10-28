import pandas as pd
import numpy as np

def impute2(df, noiseFac=0.3):
    df = df.copy()
    missing_timestamps = df[df['flows'].isna()].index

    for ts in missing_timestamps:
        past = df.loc[:ts].dropna(subset=['flows'])

        if len(past) == 0:
            continue

        val = None


        g3_mean = past.groupby(past.index.hour)['flows'].mean()
        g3_std  = past.groupby(past.index.hour)['flows'].std()

        if ts.hour in g3_mean:
            mean = g3_mean.loc[ts.hour]
            std = g3_std.loc[ts.hour]

            if pd.isna(std):
                std = 0

            noise = np.random.normal(0, std * noiseFac)
            val = mean + noise

        # # fallback
        if val is None:
            val = past['flows'].iloc[-1]

        # cant be negative
        df.loc[ts, 'flows'] = max(val, 0)

    return df