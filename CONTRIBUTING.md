# Contributing to LLMSQL

Welcome and thank you for your interest in the LLMSQL package! We welcome contributions, feedback, new issues, pull requests and appreciate your time spent with our package, and hope you find it useful!

## Important Resources

There are several places information about LLMSQL package is located:

- Our README.md files and comments. We tried to put separate README.md in all important parts of our package. We hope they will help you to understand the package better.
- We created [examples files](https://github.com/LLMSQL/llmsql-benchmark/tree/main/examples), feel free to explore them
- Our [documentation pages](https://llmsql.github.io/llmsql-benchmark/) is currently under development but we will try to serve documentation soon.

## Code Style

LLMSQL uses [ruff](https://github.com/astral-sh/ruff) for linting, [mypy](https://mypy.readthedocs.io/en/stable/) for type checking and [pdm](https://pdm-project.org/en/latest/) for dependency management.

1. Fork this repository.

2. Install [pdm](https://pdm-project.org/) (Python dependency manager):

```bash
pip3 install --user pdm
````

3. Clone your fork and install the development dependencies:

```bash
git clone https://github.com/<YOUR_USERNAME>/llmsql-benchmark.git
cd llmsql-benchmark
pdm install --without default --with dev
pre-commit install
```

> **Note**: If you see `Command 'pdm' not found`, it usually means `~/.local/bin` is not on your `PATH`.
>
> * On **Linux**, you can run:
>
>   ```bash
>   export PATH="$HOME/.local/bin:$PATH"
>   ```
>
>   (add this line to your shell config, e.g. `~/.bashrc`, `~/.zshrc`, or for fish: `set -U fish_user_paths $HOME/.local/bin $fish_user_paths`)



Great🎉! You are all set for developing!


## Testing

We use [pytest](https://docs.pytest.org/en/latest/) for running unit tests. All library unit tests can be run via:

```bash
PYTHONPATH=. pdm run pytest --cov=llmsql --cov-report=xml --maxfail=1 --disable-warnings -v
```

Please run this command before any changes, just to make sure all code you forked works well by the time the development starts.

Also to enable `pre-commit hooks` with `pre-commit install` command. Pre commit hooks contain pytest and will be run before each commit.

## Contribution Best Practices

We recommend a few best practices to make your contributions or reported errors easier to assist with.

**For Pull Requests:**

- PRs should be titled descriptively, and be opened with a brief description of the scope and intent of the new contribution.
- New features should have appropriate documentation added alongside them.
- Aim for code maintainability, and minimize code copying.

**For Feature Requests:**

- Provide a short paragraph's worth of description. What is the feature you are requesting? What is its motivation, and an example use case of it? How does this differ from what is currently supported?

**For Bug Reports**:

- Provide a short description of the bug.
- Provide a *reproducible example* - what is the command you run with our package that results in this error? Have you tried any other steps to resolve it?
- Provide a *full error traceback* of the error that occurs, if applicable. A one-line error message or small screenshot snippet is unhelpful without the surrounding context.
- Note what version of the codebase you are using, and any specifics of your environment and setup that may be relevant.




## How Can I Get Involved?

To quickly get started, we maintain a list of good first issues, which can be found by [filtering GH Issues](https://github.com/LLMSQL/llmsql-benchmark/issues?q=is%3Aopen+label%3A%22good+first+issue%22+label%3A%22help+wanted%22). These are typically smaller code changes or self-contained features which can be added without extensive familiarity with library internals, and we recommend new contributors consider taking a stab at one of these first if they are feeling uncertain where to begin.

There are a number of distinct ways to contribute to LLMSQL, and all are extremely helpful! A sampling of ways to contribute include:

- **Improving documentation** - Improvements to the documentation, or noting pain points / gaps in documentation, are helpful in order for us to improve the user experience of the package and clarity + coverage of documentation.
- **Testing and devops** - We are very grateful for any assistance in adding tests for the library that can be run for new PRs, and other devops workflows.
- **Proposing or Contributing New Features** - We want LLMSQL to be the best way to interact (evaluate) models on our benchmark. If you have a feature that is not currently supported but desired, feel free to open an issue describing the feature and, if applicable, how you intend to implement it. We would be happy to give feedback on the cleanest way to implement new functionalities and are happy to coordinate with interested contributors.

We hope that this has been helpful, and appreciate your interest in contributing!
