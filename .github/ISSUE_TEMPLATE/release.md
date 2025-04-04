---
name: New Release
about: Creating a new release version for alltheplots. Only for Maintainers.
title: "Release "
labels: "release"
assignees: "gomezzz"
---

# Feature

## Changelog

_to be written during release process_

## What Needs to Be Done (chronologically)

- [ ] Create PR to merge from current main into release branch
- [ ] Write Changelog in PR and request review
- [ ] Review the PR (if OK - merge, but DO NOT delete the branch)
- [ ] Minimize packages in requirements.txt. Update packages in setup.py
- [ ] Check unit tests -> Check all tests pass and there are tests for all important features
- [ ] Disable loguru debug logging in release version (set default log level to WARNING)
- [ ] Check documentation -> Check presence of documentation for all features
- [ ] Change version number in setup.py
- [ ] Trigger the Upload Python Package to testpypi GitHub Action (https://github.com/gomezzz/alltheplots/actions/workflows/deploy_to_test_pypi.yml) on the release branch (need to be logged in)
- [ ] Test the build on testpypi (with `pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple alltheplots`)
- [ ] Finalize release on the release branch
- [ ] Create PR: release → main
- [ ] PR Reviews
- [ ] Merge release back into main
- [ ] Create Release on GitHub from the last commit (the one reviewed in the PR)
- [ ] Upload to PyPI