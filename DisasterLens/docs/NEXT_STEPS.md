# DisasterLens AI - Next Steps

A practical, ordered plan from today's working prototype to a validated,
deployable system. Each phase has a clear **goal**, concrete **tasks** and an
**exit check**, so you know when to move on.

**Where things stand:** the dashboard, PDF reports, training / evaluation
scripts, Colab notebook and test-suite work end to end (verified on synthetic
data). What does *not* exist yet is a model trained on real xBD data - until it
does, the app runs in demo mode and the Analytics page is empty.

---

## Phase 0 - Finish the first real training run (now)

**Goal:** get one trustworthy `best_model.pth` plus its metrics.

1. In Colab run section **5a** (`--dry-run`) once to confirm paths and memory.
2. Start a short run first: `--epochs 5`. Confirm that train loss falls and
   validation mean IoU rises above zero. Only then launch the full 50 epochs.
3. If the runtime drops, re-run the setup cells and use the **resume** cell
   (5b). `last_checkpoint.pth` is written to Drive after every epoch.
4. When training ends, run section 6 (evaluate on the held-out **test** split).
5. Download `best_model.pth`, `training_history.json`, `evaluation_results.json`.
6. Install them locally (see README -> *Evaluating and connecting the model to
   the app*) and confirm the sidebar badge reads **Model Ready**.

**Exit check:** Analytics shows real curves; Image Analysis runs on a real
checkpoint (no *Demo Mode* banner).

**If results are poor** - check in this order:

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| Val loss falls, mean IoU stays near 0 | Model predicts only background | Confirm masks contain damage classes (notebook cell 11); check class sampler; train longer |
| Damaged classes (3, 4) near 0 IoU | Severe imbalance | Raise Focal weight, add class weights, oversample destroyed / major tiles |
| Train IoU high, val IoU low | Overfitting / event shift | More augmentation, fewer epochs, weight decay, more events in train |
| Loss NaN | Bad batch / LR too high | The trainer skips non-finite batches; lower LR to 5e-5 |
| Slow epochs | Data loading | `num_workers: 2` on Colab; keep the dataset on local disk (copy from Drive to `/content`) |

---

## Phase 1 - Make the model good

**Goal:** a defensible model and an honest evaluation.

### 1.1 Evaluation you can trust
- Report **per-class IoU / F1** (already produced) and add **building-level**
  metrics: connected components per class -> precision / recall of damaged
  buildings. Pixel metrics alone flatter large buildings.
- Compute the **xView2 challenge score** (0.3 x localisation F1 + 0.7 x
  damage F1) so results are comparable with published work.
- Break results down **per disaster event** - the test events are unseen, so
  this is your honest estimate of generalisation.
- Save the confusion matrix and a handful of qualitative failure cases per
  class into `docs/` for the project report.

### 1.2 Model improvements (try one at a time; keep a results table)
1. **Resolution:** train on random 512 x 512 *crops* of the native 1024 tiles
   instead of down-scaling the whole tile - small buildings keep their detail.
2. **Siamese encoder:** run pre and post through a *shared* encoder and fuse the
   feature difference, instead of stacking 6 channels (the standard strong
   baseline for change detection).
3. **Two-stage approach:** (a) building localisation (binary), (b) per-building
   damage classification on the localised footprints. This is how top xView2
   solutions separated "where are buildings" from "how damaged are they".
4. **Backbone:** ResNet-50 / EfficientNet / ConvNeXt (`model.encoder`).
5. **Loss:** tune focal / dice weights, add class weights, try Lovasz or
   Tversky for rare classes.
6. **Inference:** test-time augmentation (flips / 90-degree rotations, average
   logits) and a small ensemble.

### 1.3 Experiment hygiene
- One config per experiment (copy `config.yaml`), fixed seed, and a results
  table (config, epochs, mIoU, per-class IoU, test score).
- Add Weights & Biases or MLflow logging inside `training/train.py`.

**Exit check:** a documented table showing the baseline vs. each improvement,
and a chosen model with test-set numbers you can explain.

---

## Phase 2 - Product features

**Goal:** make the tool useful for real imagery, not just 1024-px tiles.

| Feature | Approach |
|---------|----------|
| Large images | Sliding-window inference with overlap and blending; process GeoTIFFs tile-by-tile |
| Real-world area | Read CRS / pixel size with `rasterio`; report damaged area in m^2 and hectares |
| Building-level output | Vectorise masks (`rasterio.features.shapes`); count buildings per class; export **GeoJSON / Shapefile / CSV** |
| Maps | Overlay results on a basemap (`folium` / `pydeck`) |
| Uncertainty | Show softmax confidence; highlight low-confidence regions for review |
| Human review loop | Let an analyst correct labels in the UI; store corrections for retraining |
| Batch mode | Queue several image pairs and produce one combined report |
| Report v2 | Add legend, scale, per-class building counts, uncertainty summary and analyst sign-off field |
| Image alignment | Automatic co-registration check between pre and post images; warn when mis-aligned |

Prioritise **GeoTIFF + area + GeoJSON export**: it is what GIS and
emergency-management users actually need.

---

## Phase 3 - Engineering and deployment

**Goal:** reproducible, tested, shippable.

1. **CI:** GitHub Actions running `pytest` on every push (Python 3.10 - 3.13);
   add `ruff` + `mypy` / `pre-commit`.
2. **Container:** a `Dockerfile` (python-slim, CPU torch) and
   `docker compose up`; optional GPU image.
3. **Model distribution:** do not commit `.pth` files. Publish weights as a
   GitHub Release asset or on the Hugging Face Hub and download at start-up
   (checksum-verified); keep `checkpoint_dir` pointing at the cache.
4. **Performance:** export to ONNX / TorchScript, enable half precision on
   GPU, benchmark CPU latency for a 1024 x 1024 pair (target: a few seconds).
5. **Hosting:** Streamlit Community Cloud or Hugging Face Spaces for a public
   demo (demo mode needs no weights); a small VM / container service for the
   real model.
6. **Persistence:** move history from JSON to SQLite (single user) or
   PostgreSQL (multi-user); store images in object storage.
7. **Security:** authentication for any shared deployment, upload size / type
   validation (already present), no user imagery in logs.

---

## Phase 4 - Responsible use and documentation

- **Model card:** training data, intended use, metrics (overall and per event),
  known failure modes, out-of-scope uses.
- **Domain-shift study:** evaluate on imagery that is *not* from xBD (other
  sensors / regions). Document the drop honestly.
- **Licensing:** xBD is CC BY-NC-SA 4.0. Decide whether the project and weights
  are non-commercial and state it clearly.
- **Human-in-the-loop policy:** keep the verification disclaimers; define who
  signs off on a report before it leaves the tool.
- **Report / thesis material:** architecture diagram, ablation table, failure
  gallery, ethical considerations, future work.

---

## Suggested timeline (part-time)

| Week | Focus | Deliverable |
|------|-------|-------------|
| 1 | Phase 0 | First real checkpoint; app on *Model Ready*; baseline metrics |
| 2 | Phase 1.1 + 1.2 (crops, TTA) | Evaluation script with building-level metrics; improved model |
| 3 | Phase 1.2 (Siamese or two-stage) + Phase 2 (GeoTIFF, area, GeoJSON) | Better model; geospatial exports |
| 4 | Phase 3 + 4 | CI, Docker, hosted demo, model card, final report |

## Immediate checklist

- [ ] Smoke test (`--dry-run`) passes in Colab
- [ ] 5-epoch run shows rising val mIoU
- [ ] Full run finished; test-set evaluation saved
- [ ] `best_model.pth` + JSON artifacts copied into the app
- [ ] Sidebar shows *Model Ready*; screenshot a real result for the report
- [ ] Per-event and building-level metrics added
- [ ] Decide the next model improvement (crops -> Siamese -> two-stage)
