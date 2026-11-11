from .nms import batched_nms

# REWIND (VENDORED.md, modification 3): everything except batched_nms is imported
# lazily, on first use. Inference only needs NMS; the eager imports pulled in
# evaluation/training dependencies (pandas, joblib) that the runtime doesn't ship.
_LAZY = {
    'ANETdetection': '.metrics', 'remove_duplicate_annotations': '.metrics',
    'make_optimizer': '.train_utils', 'make_scheduler': '.train_utils',
    'save_checkpoint': '.train_utils', 'AverageMeter': '.train_utils',
    'train_one_epoch': '.train_utils', 'valid_one_epoch': '.train_utils',
    'fix_random_seed': '.train_utils', 'ModelEma': '.train_utils',
    'postprocess_results': '.postprocessing',
}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        return getattr(importlib.import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ['batched_nms', 'make_optimizer', 'make_scheduler', 'save_checkpoint',
           'AverageMeter', 'train_one_epoch', 'valid_one_epoch', 'ANETdetection',
           'postprocess_results', 'fix_random_seed', 'ModelEma', 'remove_duplicate_annotations']
