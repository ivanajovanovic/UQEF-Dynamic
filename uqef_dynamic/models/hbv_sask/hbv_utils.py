from functools import partial

import dateutil.parser
import numpy as np
import pandas as pd

from common import utility

precipitation_column_name="precipitation"
temperature_column_name="temperature"
evapotranspiration_column_name="evapotranspiration"
observed_discharge = "observed_discharge"  # measured_data_column

dayfirst_parse = partial(dateutil.parser.parse, dayfirst=True)

my_local_data_path = "/dss/dssfs02/lwp-dss-0001/pr63so/pr63so-dss-0000/ga45met2/HBV-SASK-data/megala_creek_australia"

date = np.loadtxt("/dss/dssfs02/lwp-dss-0001/pr63so/pr63so-dss-0000/ga45met2/HBV-SASK-data/megala_creek_australia/date.txt",
                  converters={0: dayfirst_parse}, dtype=np.datetime64)
evapotranspiration = np.loadtxt(
    "/dss/dssfs02/lwp-dss-0001/pr63so/pr63so-dss-0000/ga45met2/HBV-SASK-data/megala_creek_australia/evapotranspiration.txt")
observed_discharge = np.loadtxt(
    "/dss/dssfs02/lwp-dss-0001/pr63so/pr63so-dss-0000/ga45met2/HBV-SASK-data/megala_creek_australia/observed_discharge.txt")
precipitation = np.loadtxt("/dss/dssfs02/lwp-dss-0001/pr63so/pr63so-dss-0000/ga45met2/HBV-SASK-data/megala_creek_australia/precipitation.txt")
temperature = np.loadtxt("/dss/dssfs02/lwp-dss-0001/pr63so/pr63so-dss-0000/ga45met2/HBV-SASK-data/megala_creek_australia/temperature.txt")

assert(len(date) == len(evapotranspiration) == len(
    observed_discharge) == len(precipitation) == len(temperature))

df = pd.DataFrame({utility.TIME_COLUMN_NAME: date, evapotranspiration_column_name: evapotranspiration, observed_discharge: observed_discharge,
    precipitation_column_name: precipitation, temperature_column_name: temperature})

print(f"df - {df}")
# df.to_pickle("/dss/dssfs02/lwp-dss-0001/pr63so/pr63so-dss-0000/ga45met2/HBV-SASK-data/megala_creek_australia/megala_creek_australia.pkl.gz")