# Architecture

## Local Foundation

Use an installable Python package under `src/panelchemy/` for the processing
core. Keep image separation, silhouette/support construction, relief generation,
validation, and export callable independently of an interface.

The command-line adapter will expose reproducible operations and configuration.
A local browser adapter will provide input selection, parameter editing, and
inspectable intermediate previews. Neither adapter should own a second copy of
the conversion algorithm.

Keep experiments separate from the core, and deliberately retained examples
separate from ignored local inputs. Select image and geometry dependencies when
implementing the relevant stage, rather than inheriting Scansor's CAD stack.

## Later Deployment

Scansor provides a structural reference: a distinct browser front end, reusable
processing behavior, and reproducible CI artifacts that can be selected for
deployment. It does not settle Panelchemy's runtime or hosting choice.

After local behavior is useful, evaluate static browser execution against a
hosted adapter using actual dependency availability, memory, performance, and
redistribution obligations. Preserve the local workflow and compare representative
results across supported runtimes. Domain selection and public application
publication are later work.
