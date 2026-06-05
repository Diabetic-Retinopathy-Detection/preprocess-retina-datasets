# preprocess-retina-datasets

[![Release](https://img.shields.io/github/v/release/Diabetic-Retinopathy-Detection/preprocess-retina-datasets)](https://img.shields.io/github/v/release/Diabetic-Retinopathy-Detection/preprocess-retina-datasets)
[![Build status](https://img.shields.io/github/actions/workflow/status/Diabetic-Retinopathy-Detection/preprocess-retina-datasets/main.yml?branch=main)](https://github.com/Diabetic-Retinopathy-Detection/preprocess-retina-datasets/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/Diabetic-Retinopathy-Detection/preprocess-retina-datasets/branch/main/graph/badge.svg)](https://codecov.io/gh/Diabetic-Retinopathy-Detection/preprocess-retina-datasets)
[![Commit activity](https://img.shields.io/github/commit-activity/m/Diabetic-Retinopathy-Detection/preprocess-retina-datasets)](https://img.shields.io/github/commit-activity/m/Diabetic-Retinopathy-Detection/preprocess-retina-datasets)
[![License](https://img.shields.io/github/license/Diabetic-Retinopathy-Detection/preprocess-retina-datasets)](https://img.shields.io/github/license/Diabetic-Retinopathy-Detection/preprocess-retina-datasets)

This is a script to preprocess retina fundus images datasets into a uniform structure.

- **Github repository**: <https://github.com/Diabetic-Retinopathy-Detection/preprocess-retina-datasets/>
- **Documentation** <https://Meier-Stefan.github.io/preprocess-retina-datasets/>

## Getting started with your project

### 1. Set Up Your Development Environment

Then, install the environment and the pre-commit hooks with

```bash
make install
```

This will also generate your `uv.lock` file

### 2. Run the pre-commit hooks

Initially, the CI/CD pipeline might be failing due to formatting issues. To resolve those run:

```bash
uv run pre-commit run -a
```

### 3. Commit the changes

Lastly, commit the changes made by the two steps above to your repository.

```bash
git add .
git commit -m 'Fix formatting issues'
git push origin main
```

You are now ready to start development on your project!
The CI/CD pipeline will be triggered when you open a pull request, merge to main, or when you create a new release.

To finalize the set-up for publishing to PyPI, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/publishing/#set-up-for-pypi).
For activating the automatic documentation with MkDocs/Zensical, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/docs_tool/#deploying-to-github-pages).
To enable the code coverage reports, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/codecov/).


---

Repository initiated with [osprey-oss/cookiecutter-uv](https://github.com/osprey-oss/cookiecutter-uv).
