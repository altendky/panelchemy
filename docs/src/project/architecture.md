# Architecture

## Local Foundation

Use an installable Python package under `src/panelchemy/` for the processing
core. Keep image separation, silhouette/support construction, relief generation,
validation, and export callable independently of an interface.

Use OpenCascade for solid modeling, following Scansor's kernel choice, through
Python bindings such as OCP. Keep the binding/dependency selection bounded to
geometry implementation. OCCT provides
[planar wire offsets](https://dev.opencascade.org/doc/refman/html/class_b_rep_offset_a_p_i___make_offset.html)
and [Boolean/splitting operations](https://dev.opencascade.org/doc/overview/html/specification__boolean_operations.html)
for feature construction and material bands. Evaluate robustness and performance
on many small contours before expanding to full artworks.

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
