Whenever you are committing a Jupyter notebook, please use `nbstripout` to automatically ignore output when committing.\
Install once:
```bash
pip install nbstripout
```
Activate in your Git repo:
```bash
nbstripout --install
```
Now, whenever you commit a notebook, outputs will be stripped automatically.\
Otherwise, please clear outputs directly in your local file, either from Jupyter GUI or bash
```bash
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace your_notebook.ipynb
```
After this, commit and push as usual.
