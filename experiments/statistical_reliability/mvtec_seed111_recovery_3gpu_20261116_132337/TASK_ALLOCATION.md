# Seed111 remaining-category allocation

Current user requested distribution across three GPUs. Existing10 completed categories remain untouched. No additional seeds started.

| Physical GPU | Sequential tasks | Training batches remaining |
|---|---|---:|
|0|tile resume completed epoch15→60, then transistor epochs1–60|2925|
|1|wood epochs1–60, then toothbrush epochs1–60|2340|
|2|zipper epochs1–60|1800|

1022×1022, Batch8, LR5e-4, total60 epochs, seed111, FP32. Independent CUDA_VISIBLE_DEVICES UUIDs each expose logical GPU0. This preserves the one-device CUDA RNG contract for the tile checkpoint. Exact parameters/data/code identity remain enforced by the original resume path. Strict bitwise training replay is not guaranteed by the existing backward backend.

Each category has a new artifacts directory, category.log, command.json, first_step.json and exit_status.json. The first tile optimizer step must be436 (not1). Three GPU group workers run concurrently, tasks within each group run sequentially. Failure stops that group without terminating other groups or reducing batch.

Approximate additional checkpoint space245GB; free space checked at3.7TB before launch. Prior ENOSPC root cause beyond the recorded storage error is not certified; all snapshots are preserved and no automatic deletion is done.

On successful completion, SEED111_COMPLETE.json and RESULTS.md consolidate the existing10 and new5 complete category exports, requiring15 categories/1725 test images. No old result is overwritten. No five-seed Mean±SD or bootstrap is performed. PAPER REPRODUCTION UNVERIFIED.
