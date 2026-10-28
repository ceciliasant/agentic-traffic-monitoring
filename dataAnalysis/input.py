import pandas as pd
import numpy as np

def impute(df, noiseFac=0.3):
    df = df.copy()
    missing_timestamps = df[df['flows'].isna()].index

    for ts in missing_timestamps:
        past = df.loc[:ts].dropna(subset=['flows'])

        if len(past) == 0:
            continue

        val = None

        # weekday + hour
        g2_mean = past.groupby([past.index.weekday, past.index.hour])['flows'].mean()
        g2_std  = past.groupby([past.index.weekday, past.index.hour])['flows'].std()

        key2 = (ts.weekday(), ts.hour)

        if key2 in g2_mean:
            mean = g2_mean.loc[key2]
            std = g2_std.loc[key2]

            if pd.isna(std):
                std = 0

            noise = np.random.normal(0, std * noiseFac)
            val = mean + noise

        # if no average weekday + hour exists, use average of hour
        if val is None:
            g3_mean = past.groupby(past.index.hour)['flows'].mean()
            g3_std  = past.groupby(past.index.hour)['flows'].std()

            if ts.hour in g3_mean:
                mean = g3_mean.loc[ts.hour]
                std = g3_std.loc[ts.hour]

                if pd.isna(std):
                    std = 0

                noise = np.random.normal(0, std * noiseFac)
                val = mean + noise

        # in case no average weekday + hour or hour exists
        if val is None:
            val = past['flows'].iloc[-1]

        # >0
        df.loc[ts, 'flows'] = max(val, 0)

    return df