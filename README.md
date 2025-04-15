# MaskBench

This repository contains the code, experiments and documentation for the master's project "MaskBench -  A Comprehensive Benchmark Framework for Video De-identification". The project is supervised by Prof. Gerard de Melo at Hasso-Plattner-Institute (HPI) during the summer term 2025.

## Installation Instructions
1. Install [Docker](https://www.docker.com/) and make sure the Docker daemon is running.
2. Run `docker-compose build` to build the docker container from the provided image. This must only be done once or whenever
   - changing the `Dockerfile`
   - changing the `docker-compose.yml`
   - you add dependencies and change the `pyproject.toml` or `poetry.lock`
3. Copy the env files `cp .env.dist .env`.
4. Open the .env file with `vim .env` or `nano .env`. Inside this file we need to specify the absolute path to the `MASKBENCH_DATASET_DIR` and the `MASKBENCH_OUTPUT_DIR`. If you do not have an input and/or output folder, you can create them either within the maskbench repository or outside with `mkdir input && mkdir output` and add the absolute path to the .env file.
5. Run `docker-compose up` to run the docker container. You need to execute this command whenever you make changes to the code base.

## Commit Guideline
We use the [Conventional Commits Specification v1.0.0](https://www.conventionalcommits.org/en/v1.0.0/#summary) for writing commit messages. Refer to the website for instructions.

### Commit Types

We use the recommended commit types from the specification, namely:

- `feat:` A code change that introduces a **new feature** to the codebase (this correlates with MINOR in Semantic Versioning)
- `fix:` A code change that **patches a bug** in your codebase (this correlates with PATCH in Semantic Versioning)
- `refactor:` A code change that **neither fixes a bug nor adds a feature**
- `build:` Changes that **affect the build system** or external dependencies (example scopes: pip, npm)
- `ci:` Changes to **CI configuration** files and scripts (examples: GitHub Actions)
- `docs:` **Documentation only** changes
- `perf:` A code change that **improves performance**
- `test:` Adding missing **tests** or correcting existing tests

### How should I voice the commit message?

- `feat:` commits: use the imperative, present tense – eg. `feat: add button` not `feat: added button` nor `feat: adds button`
- `fix:` commits: describe the bug that is being fixed – eg. `fix: button is broken` not `fix: repair broken button`

### What if I introduce a breaking change?

- Option 1): include an exclamation mark (`!`) after the commit type to draw attention to a breaking change
```
feat!: send an email to the customer when a product is shipped
```
- Option 2): include a breaking change footer
```
feat: allow provided config object to extend other configs

BREAKING CHANGE: `extends` key in config file is now used for extending other config files
```

### What do I do if the commit conforms to more than one of the commit types?

Go back and make multiple commits whenever possible. Part of the benefit of Conventional Commits is its ability to drive us to make more organized commits and PRs.

