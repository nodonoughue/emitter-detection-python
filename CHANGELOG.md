# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-12

### Added
- `ewgeo.tracker` — full multi-target tracking subsystem:
  - Kalman filter state representation (`State`, `StateSpace`, `Track`)
  - Motion models: Constant Velocity, Constant Acceleration, Constant Jerk
  - Associators: Nearest-Neighbor (NN), Global Nearest-Neighbor (GNN), Probabilistic Data
    Association (PDA)
  - Initiators: Single-Point, Two-Point (velocity estimate), Three-Point (accel estimate)
  - M-of-N promoter and missed-detection deleter
  - `Tracker` orchestration class with optional live plotting
  - Comprehensive pytest test suite (`tests/test_tracker_*.py`)
- `pyproject.toml`: `[project.optional-dependencies] dev` group and
  `[tool.pytest.ini_options]` section for zero-config `pytest` runs
- `CHANGELOG.md` (this file)

### Changed
- `PassiveSurveillanceSystem` subclasses (`TDOAPassiveSurveillanceSystem`,
  `FDOAPassiveSurveillanceSystem`, `HybridPassiveSurveillanceSystem`) moved to dedicated
  `system.py` files within each geolocation subpackage
- Docstrings added or corrected throughout: `tracker/`, `utils/covariance.py`,
  `utils/search_space.py`, `utils/unit_conversions.py`, `utils/utils.py`,
  `atm/reference.py`, `array_df/solvers.py`, `atm/model.py`
- `tracker/__init__.py`: replaced wildcard imports with explicit named imports
- `make_figures/atm_itu_validation.py`: moved from `src/ewgeo/atm/test.py` so it is
  no longer shipped as part of the installed package

### Fixed
- `ConstantJerkMotionModel.make_transition_matrix` and
  `make_process_covariance_matrix` docstrings incorrectly said "constant acceleration"
- `utils/geo.py:find_intersect` — `:param psi1` description referenced wrong variable
- `utils/utils.py:make_prior` — `:return fim_prior` incorrectly described return value
  as a callable; it is a constant ndarray
- README: `ewgeo.array` → `ewgeo.array_df`; "Appendix Carlo" → "Appendix C";
  added missing `ewgeo.tracker` entry

## [0.0.1] - initial release

- Port of MATLAB companion code for *Emitter Detection and Geolocation for Electronic
  Warfare* (Artech House, 2019) and *Practical Geolocation for Electronic Warfare using
  MATLAB* (Artech House, 2022)
- Modules: `aoa`, `array_df`, `atm`, `detector`, `fdoa`, `hybrid`, `noise`, `prop`,
  `tdoa`, `triang`, `utils`
