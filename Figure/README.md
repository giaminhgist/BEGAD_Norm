# Figure

All figure-generation code and figure outputs requested for this project
live here.

| Path | Content |
|---|---|
| `code/` | reproducible figure scripts (one script per figure; keep the data-source path, export format, and dimensions in the script header) |
| `outputs/` | exported figures (SVG/PDF preferred, PNG when raster is required) |
| `Figure_text.md` | caption + description of every figure; update it each time a new figure is created |

Conventions:

- Figure code reads real result/data files only (never hard-codes numbers
  into plots).
- Exported files are named `fig_<topic>.<ext>` and match the script name
  (`code/fig_<topic>.py` → `outputs/fig_<topic>.svg`).
- A figure is complete when the rendered export has been inspected and the
  plotted series verified against the source data.
