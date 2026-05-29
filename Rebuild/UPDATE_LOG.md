# Update Record

## 2026-05-21 20:24

Files modified:
- Rebuild/fssdetector_legacy.py
- Rebuild/fssdetector_pipeline.py
- Rebuild/FSS_DETECTOR_PIPELINE_STRUCTURE.txt
- Rebuild/UPDATE_LOG.md

Changes:
- Added component geometry extraction for local YOLO patches: area, aspect ratio, fill ratio, solidity, elongation, skeleton length, estimated stroke width, border contact, and local contrast.
- Added conservative annotation component classification for arrow, line, and text candidates.
- Refactored patch mask refinement so YOLO boxes only localize candidate regions; final masks are connected-component masks, not bbox masks.
- Added structure-aware repair gating in process_edges(...).
- Split removed annotation pixels from repair-needed pixels.
- Added background-only region handling: fill background-only removed pixels white instead of inpainting them.
- Added debug masks: refined_component_mask.png, structure_mask.png, gated_repair_mask.png, background_only_regions.png, *_repair_needed_mask.png, and *_dilated_structure_mask.png.
- Kept diff_annotation_mask as debug output only; it is not merged into the final repair mask.

Reason:
- Reduce over-removal from YOLO detections and prevent background labels/arrows/lines from being inpainted as if they were damaged FSS structure.

Expected impact:
- Arrow/text bboxes should no longer erase large surrounding structure.
- Margin text and background annotation lines should be removed or whitened without inpainting artifacts.
- Structure-touching annotations should still be repaired.
- Downstream edge extraction, contour detection, B-spline fitting, and CST parameterization should receive cleaner geometry.

Compatibility:
- Public API remains unchanged.
- Existing result keys remain available.
- The existing "mask" key now represents the final gated repair-needed mask.
- Additional debug/result keys were appended safely.

Known risks:
- Conservative component classification may miss some faint or thick annotations.
- Structure mask estimation can still be sensitive to unusual color clustering.
- Background-only fill assumes the normalized downstream canvas should be white.

## 2026-05-21 21:40

Files modified:
- Rebuild/FssDetector.py
- Rebuild/fssdetector_selection.py
- Rebuild/fssdetector_pipeline.py
- Rebuild/FSS_DETECTOR_PIPELINE_STRUCTURE.txt
- Rebuild/UPDATE_LOG.md

Changes:
- Added yolo_element_conf_threshold with default 0.12 for arrow/line detections.
- Kept yolo_conf_threshold at 0.20 for text/subfigure detection.
- Added repair_structure_dilation_pixels with default 10 for structure-aware repair gating.
- Added optional conf_threshold argument to detect_elements_with_yolo(...).
- remove_detected_elements_with_yolo(...) now applies the lower element threshold for arrow/line detection.
- process_edges(...) now reads repair_structure_dilation_pixels instead of using a hard-coded 6 px repair gate dilation.

Reason:
- Some annotation pixels that should become repair-needed masks were missing after the structure-aware update. The likely causes were low arrow/line recall and an overly tight structure-proximity gate.

Expected impact:
- Better recall for faint/small arrow and line annotations.
- More structure-touching annotation masks should pass into repair_needed_mask.
- Text skip guard remains stable because text still uses yolo_conf_threshold.

Compatibility:
- Public API remains compatible.
- New constructor arguments are optional and have defaults.
- Existing output fields are unchanged.

Known risks:
- Lower arrow/line confidence may add more false-positive candidates, but component refinement and structure gating should prevent most background-only detections from becoming inpaint masks.
- A larger structure gate may include slightly more near-structure annotation pixels, which is intentional but should be monitored on dense designs.
