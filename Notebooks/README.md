Whenever you are committing a Jupyter notebook, please use `nbstripout` to automatically ignore output when committing.\
Install once:
```bash
python -m pip install --upgrade nbstripout
pip install nbstripout
```
Activate in your Git repo:
```bash
nbstripout --install
```
Make sure it runs:
```bash
git config --local --get filter.nbstripout.clean
```
Now, whenever you commit a notebook, outputs will be stripped automatically.\
Otherwise, please clear outputs directly in your local file, either from Jupyter GUI or bash
```bash
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace your_notebook.ipynb
```
After this, commit and push as usual.

- Notebook `HBV_data_analysis.ipynb` contains code snippets for reading data for two basins for the HBV-SASK model, for simple processing and plotting of the data
- Notebook `HBV_model_runs.ipynb` contains code snippets for simply running the model and examing/plotting the output

