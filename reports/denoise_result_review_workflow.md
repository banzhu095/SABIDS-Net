# Denoising result review workflow

The workflow has two independent phases. Juchiyun discovers the latest structurally and numerically complete locked run, summarizes method provenance, and packages lossless PKU37 test assets by anatomical position. Windows verifies the downloaded package, records blinded square ROIs using noisy/reference images only, locks and versions those coordinates, and then applies identical crops to every available method.

Full-image benchmark results remain the primary test evidence. A fixed-ROI analysis is confirmatory only when every PKU37 test position is represented; smaller selections are labelled `exploratory_descriptive` and produce no significance p-values. Bootstrap confidence intervals resample anatomical positions, never ROIs, repeat frames, or model seeds.

Primary seeds come from the locked inference registry. ROI results cannot select a checkpoint or seed. Missing assets remain missing, and qualitative panels show `MISSING` rather than substituting another image.

See `tools/denoise_result_review/README.md` for exact cloud packaging, recovery, Windows selection, evaluation, workbook, and all-seed sensitivity commands.
