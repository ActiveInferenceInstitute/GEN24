# ARCANGEL notebook

`ARCANGEL1_0_2.ipynb` is a two-cell Google Colab notebook: a Colab badge/overview cell and one large Python code cell.

The code cell:

- mounts Google Drive at `/content/drive`;
- imports scientific and machine-learning libraries including NumPy, SciPy, scikit-learn, TensorFlow, PyTorch, NetworkX, scikit-image, tqdm, pandas, and Matplotlib;
- configures random seeds and CPU behavior;
- defines preprocessing utilities and cognitive/problem-solving classes; and
- loads training/evaluation challenge data before running an evaluation flow.

The notebook prints library versions and evaluation progress. It should be opened in Google Colab or another environment with the same dependencies and access to the expected Drive-mounted data. The repository does not include a requirements file specifically for this notebook or the challenge data it expects.

The notebook is independent of the `GEN-24/` file pipeline and the `NHIM` module. The top-level README links to the Colab-compatible notebook file directly.
