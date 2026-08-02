# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

## [1.4.0] - 2025-08-02

### Added
- AGENTS.md with development guidelines for AI assistants

### Changed
- **Project renamed**: Heretic -> Model Censorship Removal (MCR)
- Module: `src/heretic/` -> `src/mcr/`
- Package: `heretic-llm` -> `model-censorship-removal`
- CLI: `heretic` -> `mcr`
- **README.md rewritten**: Professional, technical documentation targeting enterprise and research audiences
- Updated all Python imports and plugin paths across the codebase
- Updated configuration files and test configs to reflect new module paths
- Added Albert Mendes as project author alongside Philipp Emanuel Weidmann

### Removed
- pip/uv distribution support (git clone only)

### Preserved
- Optuna study name `"heretic"` (internal checkpoint identifier, required for compatibility)
- HuggingFace model tag `"heretic"` (community standard, 4000+ published models use it)
- Model suffix `-heretic` (published model naming convention)
- Original Heretic citation and upstream references
