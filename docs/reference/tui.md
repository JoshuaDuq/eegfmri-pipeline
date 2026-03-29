# TUI Reference

This is the technical reference for the optional Go terminal UI that wraps the public CLI.

## Purpose

The TUI provides guided command construction, live execution monitoring, and persistent
project state without changing the underlying CLI behavior.

## Main Views

- main menu
- pipeline wizard
- execution view
- global setup
- dashboard
- history
- quick actions
- pipeline smoke test

## Architecture

The application is a single Bubble Tea program with:

- a root model and navigation stack
- view-specific update and render logic
- a subprocess executor that launches Python commands from the repo root
- persistent state for overrides and history

## Persistence

State is stored under derivatives and TUI cache locations so repeated sessions can reuse
recent pipeline selections and project overrides.

## Keyboard Model

The UI supports:

- global navigation and quit handling
- wizard selection and editing shortcuts
- execution-log scrolling and reruns
- dashboard and history refresh flows
