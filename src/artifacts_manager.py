import logging
import torch
import tarfile
import os
import tempfile
import json
import typing

from .cvt_model import CVTModel

class ArtifactsManager(object):
    """An objects for loading training artifacts file.

    Arguments
    ---------
    save_to : str
        Path to the folders for saving the model and artifacts.
    device : int
        GPU device to use. Set to -1 for CPU.
    logger : `~logging.Logger`
        Logger object to use for logging.
    """

    MODEL_METADATA_FILENAME = 'model_metadata.json'
    MODEL_PARAMS_FILENAME = 'model_params.pth'
    WORDS_FIELD_NAME = 'words'
    CHARS_FIELD_NAME = 'chars'
    TAGS_FIELD_NAME = 'tags'

    def __init__(self,
                 save_to: str,
                 device: object = None,
                 logger: typing.Optional[logging.Logger] = None) -> None:
        if logger is None:
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(levelname)s - %(name)s - %(message)s'))
            logger.addHandler(handler)

        self.save_to = save_to
        self.device = device
        self.logger = logger

        ## initialize ##
        self._prepare_for_serialization()

    def _prepare_for_serialization(self) -> None:
        self.logger.info('Preparing serialization directory in %s', self.save_to)
        os.makedirs(self.save_to, exist_ok=True)
        self.model_metadata_path = os.path.join(self.save_to, self.MODEL_METADATA_FILENAME)
        self.model_params_path = os.path.join(self.save_to, self.MODEL_PARAMS_FILENAME)
        self.artifacts_path = os.path.join(self.save_to, 'artifacts.tar.gz')
        self.artifact_paths = [self.model_metadata_path, self.model_params_path]

    def save_model_and_artifacts(self, model) -> None:
        # prepare model's information to be saved #
        model_args = (model.num_words, model.num_tags)
        model_kwargs = dict(
            num_chars = model.num_chars,
        )

        self.logger.info('Saving model metadata to %s', self.model_metadata_path)
        with open(self.model_metadata_path, 'w') as f:
            json.dump({'args': model_args, 'kwargs': model_kwargs}, f, indent=2, sort_keys=True)
        self.logger.info('Saving model parameters to %s', self.model_params_path)
        torch.save(model.state_dict(), self.model_params_path)
        self.logger.info('Saving training artifacts to %s', self.artifacts_path)
        with tarfile.open(self.artifacts_path, 'w:gz') as tar:
            for path in self.artifact_paths:
                tar.add(path, arcname=os.path.basename(path))

    def load_model_and_artifacts(self):
        self.logger.info('Loading artifacts from %s', self.artifacts_path)
        artifact_names = [
            self.MODEL_METADATA_FILENAME,
            self.MODEL_PARAMS_FILENAME,
        ]

        with tempfile.TemporaryDirectory() as tmpdirname:
            self.logger.info('Extracting artifacts to %s', tmpdirname)
            with tarfile.open(self.artifacts_path, 'r:gz') as f:
                members = [member for member in f.getmembers()
                           if member.name in artifact_names]
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(f, tmpdirname, members=members)

            self.logger.info('Loading model metadata')
            with open(os.path.join(tmpdirname, self.MODEL_METADATA_FILENAME)) as fm:
                self.model_metadata = json.load(fm)

            self.logger.info('Building model and restoring model parameters')
            model = CVTModel(
                *self.model_metadata['args'], **self.model_metadata['kwargs'])
            model.initialize()
            # Load to CPU, see https://discuss.pytorch.org/t/on-a-cpu-device-how-to-load-checkpoint-saved-on-gpu-device/349/4  # noqa
            model.load_state_dict(
                torch.load(os.path.join(tmpdirname, self.MODEL_PARAMS_FILENAME),
                           map_location=lambda storage, loc: storage))
            model.set_device(self.device)
            # model.cuda(self.device)

        return model


