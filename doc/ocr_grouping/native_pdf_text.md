# Native PDF Text Mode

PDF OCR remains raster-first by default. Set
`KG_DOC_OCR_PDF_TEXT_MODE=prefer_native` to use embedded, positioned PDF text
when it is safe to do so, and to fall back to the configured OCR provider for
scanned pages or pages containing embedded raster images:

```text
KG_DOC_OCR_PDF_TEXT_MODE=prefer_native
```

Native runs are exported as line-level `TextCluster` records in PDF-point
coordinates and declare `spatial_accuracy="line"`. They are not promoted to
character-level grounding. Pages without positioned native text, or with
embedded images, remain on the OCR path so figures and scanned content are not
silently discarded.

`native_only` is available for validation and controlled deployments. It fails
closed when a page cannot be represented by native text rather than accepting
an ungrounded empty result. The selected mode is part of the OCR configuration
fingerprint and page provenance, so changing it cannot reuse stale artifacts.

This first implementation does not infer table structure or reconcile native
text with partial OCR regions. Independent table/figure completeness and
hybrid region reconciliation remain release-gated work in B13/C06.
