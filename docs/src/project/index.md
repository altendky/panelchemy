# Panelchemy Project

Panelchemy will convert user-supplied graphic artwork into reproducible,
configurable layered relief. Start with a local workflow; later deployment may
follow Scansor's browser/deployment structure if the processing runtime fits.

## Current Status

The repository foundation and the image-space separation API/CLI are implemented.
Separation produces reviewable palette, silhouette, visible-color, and cumulative
support masks. Relief generation, the browser application, printable geometry,
and physical printing remain subsequent stages.

The handoff's animal panels are evaluation inputs. Their exact artwork details
are not requirements for the general tool. The broader art-generation prompt
workflow is separate from the initial conversion work.

## Documentation

- [Construction requirements and open choices](requirements.md)
- [Architecture](architecture.md)
- [Development and automation](development.md)
- [Work stages](stages.md)
- [Color separation workflow](separation.md)
