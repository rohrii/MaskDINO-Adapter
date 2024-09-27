import os
from collections import OrderedDict
from urllib.parse import urlparse
from typing import IO, Any, Dict, List, Optional, cast

import detectron2.utils.comm as comm
import torch
from detectron2.checkpoint import DetectionCheckpointer
from torch.nn.parallel import DistributedDataParallel
from fvcore.common.checkpoint import _IncompatibleKeys


class FinetuningCheckpointer(DetectionCheckpointer):
    def __init__(self, model, save_dir="", *, save_to_disk=None, ignore_fix: list[str] = [], **checkpointables):
        super().__init__(model, save_dir, save_to_disk=save_to_disk, **checkpointables)
        self.ignore_fix = ignore_fix

    def save(self, name: str, **kwargs: Any) -> None:
        """
        Dump model and checkpointables to a file.

        Args:
            name (str): name of the file.
            kwargs (dict): extra arbitrary data to save.
        """
        if not self.save_dir or not self.save_to_disk:
            return

        data = dict(model=self._get_tuned_state_dict())
        for key, obj in self.checkpointables.items():
            data[key] = obj.state_dict()
        data.update(kwargs)

        basename = "{}.pth".format(name)
        save_file = os.path.join(self.save_dir, basename)
        assert os.path.basename(save_file) == basename, basename

        self.logger.info("[FinetuningCheckpointer] Saving checkpoint to {}".format(save_file))
        
        with self.path_manager.open(save_file, "wb") as f:
            torch.save(data, cast(IO[bytes], f))
        
        self.tag_last_checkpoint(basename)

    def resume_or_load(self, path: str, *, resume: bool = True) -> Dict[str, Any]:
        """
        If `resume` is True, this method attempts to resume from the last
        checkpoint, if exists. Otherwise, load checkpoint from the given path.
        This is useful when restarting an interrupted training job.

        Args:
            path (str): path to the checkpoint from cfg.
            resume (bool): if True, resume from the last checkpoint if it exists
                and load the model together with all the checkpointables. Otherwise
                only load the model without loading any checkpointables.

        Returns:
            same as :meth:`load`.
        """
        if resume and self.has_checkpoint():
            self.load(path, base_weights=True, checkpointables=[])
            path = self.get_checkpoint_file()
            return self.load(path, base_weights=False)
        else:
            return self.load(path, base_weights=True, checkpointables=[])

    def load(self, path: str, *, base_weights: bool, checkpointables: Optional[List[str]] = None):
        ##### START PARENT ######
        assert self._parsed_url_during_load is None
        need_sync = False

        if path and isinstance(self.model, DistributedDataParallel):
            path = self.path_manager.get_local_path(path)
            has_file = os.path.isfile(path)
            all_has_file = comm.all_gather(has_file)
            if not all_has_file[0]:
                raise OSError(f"File {path} not found on main worker.")
            if not all(all_has_file):
                self.logger.warning(
                    f"Not all workers can read checkpoint {path}. "
                    "Training may fail to fully resume."
                )
                # TODO: broadcast the checkpoint file contents from main
                # worker, and load from it instead.
                need_sync = True
            if not has_file:
                path = None  # don't load if not readable

        if path:
            parsed_url = urlparse(path)
            self._parsed_url_during_load = parsed_url
            path = parsed_url._replace(query="").geturl()  # remove query from filename
            path = self.path_manager.get_local_path(path)

        ##### START GRAND PARENT ######
        if not path:
            # no checkpoint provided
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        
        self.logger.info(f"[FinetuningCheckpointer] Loading {'base' if base_weights else 'extra'} weights from {path} ...")
        if not os.path.isfile(path):
            path = self.path_manager.get_local_path(path)
            assert os.path.isfile(path), "Checkpoint {} not found!".format(path)

        checkpoint = self._load_file(path)
        incompatible = self._load_model(checkpoint) if base_weights else self._load_tuned_state_dict(checkpoint)      

        if incompatible is not None:
            self._remove_expected_incompatible_keys(incompatible, base_weights)
            # This does nothing if every list is empty
            self._log_incompatible_keys(incompatible)
        
        for key in self.checkpointables if checkpointables is None else checkpointables:
            if key in checkpoint:
                self.logger.info("Loading {} from {} ...".format(key, path))
                obj = self.checkpointables[key]
                obj.load_state_dict(checkpoint.pop(key))
        ##### END GRAND PARENT ######

        if need_sync:
            self.logger.info("Broadcasting model states from main worker ...")
            self.model._sync_params_and_buffers()
        self._parsed_url_during_load = None  # reset to None
        ##### END PARENT ######
        
        return checkpoint
    
    def _load_tuned_state_dict(self, state_dict: Dict[str, Any]) -> None:
        state_dict = state_dict["model"]
        
        missing_keys = []
        unexpected_keys = []
        incorrect_shapes = []

        model_state = self.model.state_dict()

        for name, param in state_dict.items():
            if name not in model_state:
                unexpected_keys.append(name)
                continue

            if param.shape != model_state[name].shape:
                incorrect_shapes.append((name, param.shape, model_state[name].shape))
                continue

            if name in model_state:
                # Copy the parameter to the model
                model_state[name].copy_(param)

        for name in model_state:
            if name not in state_dict:
                missing_keys.append(name)

        return _IncompatibleKeys(
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
            incorrect_shapes=incorrect_shapes,
        )
        
    def _get_tuned_state_dict(self) -> Dict[str, Any]:
        state = self.model.state_dict()
        ret = OrderedDict()

        for key, value in state.items():
            # Add only the keys that we care about
            for ignore in self.ignore_fix:
                if ignore in key:
                    ret[key] = value

        return ret
    
    def _remove_expected_incompatible_keys(self, incompatible: _IncompatibleKeys, base_weights: bool) -> None:
        if base_weights:
            missing = incompatible.missing_keys.copy()
            
            # Suppress warnings about missing tunable keys
            for key in missing:
                for ignore in self.ignore_fix:
                    if ignore in key:
                        incompatible.missing_keys.remove(key)
            return

        unexpected_keys = []
        
        # Only keep tunable keys, as they should be present in the fine-tune checkpoint
        for key in incompatible.missing_keys:
            for ignore in self.ignore_fix:
                if ignore in key:
                    unexpected_keys.append(key)

        incompatible.missing_keys.clear()
        incompatible.missing_keys.extend(unexpected_keys)