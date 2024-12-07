"""
This code/data and model are from 
@Article{egusphere-2023-2865,
AUTHOR = {Mingo, D. N. and Nijzink, R. and Ley, C. and Hale, J. S.},
TITLE = {Selecting a conceptual hydrological model using Bayes' factors computed with Replica Exchange Hamiltonian Monte Carlo and Thermodynamic Integration},
JOURNAL = {EGUsphere},
VOLUME = {2024},
YEAR = {2024},
PAGES = {1--45},
URL = {https://egusphere.copernicus.org/preprints/2024/egusphere-2023-2865/},
DOI = {10.5194/egusphere-2023-2865}
}
"""
from functools import partial

import dateutil.parser
import numpy as np
import pandas as pd

from uqef_dynamic.utils import utility

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