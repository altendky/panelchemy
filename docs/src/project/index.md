# Panelchemy Project

Panelchemy will convert user-supplied graphic artwork into reproducible,
configurable layered relief. Start with a local workflow; later deployment may
follow Scansor's browser/deployment structure if the processing runtime fits.

## Current Status

The repository foundation, image-space separation API/CLI, and local browser
workflow are implemented. They produce reviewable palette, silhouette,
visible-color, and cumulative support masks. The OCCT relief API/CLI additionally
builds exact layered solids and projected support previews. Validated slicer
exports and physical printing remain subsequent stages.

The handoff's animal panels are evaluation inputs. Their exact artwork details
are not requirements for the general tool. The broader art-generation prompt
workflow is separate from the initial conversion work.

## Documentation

- [Construction requirements and open choices](requirements.md)
- [Architecture](architecture.md)
- [Development and automation](development.md)
- [Work stages](stages.md)
- [Color separation workflow](separation.md)
- [Local browser workflow](local-browser.md)
- [Layered relief construction](relief.md)
