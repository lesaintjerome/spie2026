# How to contribute

## Getting started

From the project folder:

- Create the Python environment used by binder:
  ```bash
  conda env create --name spie2026 --file .binder/environment.yml
  ```
- Activate it:
  ```bash
  conda activate spie2026
  ```
- Start jupyterlab:
  ```bash
  jupyter lab
  ```

## Python environment update

To update the Python environment used by binder:

- Update the `packages.yml` environment file
- Create a new environment from this file:
  ```bash
  conda env create --name spie2026 --file packages.yml
  ```
- Activate it:
  ```bash
  conda activate spie2026
  ```
- Export the environment configuration:
  ```bash
  conda env export > .binder/environment.yml
  ```

WARNING: the export command may generate a environment.yml file with constraints that are specific to the local platform and which may have the docker on binder fail to install... It seems recommended to have 
```
- pip
- python=3.12 # e.g.
- setuptools
- wheels
+ pip deps
```
rather than pip=3.34=hsd32=52khdfnm (e.g.).

## Useful links

- [Binder documentation](https://mybinder.readthedocs.io/en/latest/)
- https://mybinder.org/
