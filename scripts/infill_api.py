import os
import json
import tempfile
import _rs_utils as pgrsu
from io import StringIO
from abc import abstractmethod


class InfillAPI:
    @abstractmethod
    def infill(self, input: str, top_k: int):
        raise NotImplementedError("To be implemented")

    @abstractmethod
    def commit(self) -> list[list[str]]:
        raise NotImplementedError("To be implemented")

    @abstractmethod
    def start(self):
        raise NotImplementedError("To be implemented")

    @abstractmethod
    def stop(self):
        raise NotImplementedError("To be implemented")

    @classmethod
    def make(cls, **kwargs):
        return cls(**kwargs)


class ServiceBasedInfillAPI(InfillAPI):
    def __init__(self, **kwargs):
        self.__args = kwargs
        self.__inputs = []

    def start(self):
        # Start the http service in other process
        import http
        import time
        import requests
        import subprocess

        self.__proc = subprocess.Popen(
            ["bash", "-c", self._cmd_fmt.format(**self.__args)],
            stdout=subprocess.PIPE if os.getenv("DEBUG") != "1" else None,
            stderr=subprocess.PIPE if os.getenv("DEBUG") != "1" else None,
        )

        heath_api = "http://localhost:37654/health"
        pgrsu._ilog(f"Waiting for {self.__class__.__name__} to be ready")
        while True:
            time.sleep(5)
            try:
                if requests.post(heath_api).status_code == http.HTTPStatus.OK:
                    break
            except Exception:
                pgrsu._wlog(f"{self.__class__.__name__} is not ready")
                if self.__proc.poll() is not None:
                    pgrsu._flog(
                        "The process is terminated\n",
                        "stderr:\n",
                        self.__proc.stderr.read().decode(),
                    )
        pgrsu._ilog(f"{self.__class__.__name__} is ready")

    def stop(self):
        import requests

        pgrsu._ilog(f"Waiting for {self.__class__.__name__} to stop")
        exit_api = "http://localhost:37654/exit"
        requests.post(exit_api)
        self.__proc.wait()
        pgrsu._ilog(f"{self.__class__.__name__} is stopped")

    def current_num_infill(self) -> int:
        return len(self.__inputs)

    def infill(self, input: str, top_k: int):
        assert top_k == 1
        self.__inputs.append(input)

    def commit(self) -> list[list[str]]:
        assert len(self.__inputs) > 0
        outputs = []
        # Request the http API to get the outputs

        import http
        import requests

        url = "http://localhost:37654/inference"
        headers = {"Content-Type": "application/json"}
        data = json.dumps(self.__inputs)
        response = requests.post(url, headers=headers, data=data)
        if response.status_code != http.HTTPStatus.OK:
            pgrsu._flog(
                "Failed to request the http API",
                response.text,
                exp=RuntimeError("Failed to request the http API"),
            )

        outputs = json.loads(response.text)
        self.__inputs = []
        return outputs


class FinetunedUniXcoderInfillAPI(ServiceBasedInfillAPI):
    _cmd_fmt = """\
IM4DNN_REMOVE_ROOT=1 python -u -W ignore scripts/finetune_unixcoder.py \
    --run_inference_service \
    --model_name_or_path "{model_name_or_path}" \
    --output_dir "{output_dir}" \
    --max_source_length {max_source_length} \
    --max_target_length {max_target_length} \
    --beam_size {beam_size} \
    --train_batch_size {train_batch_size} \
    --eval_batch_size {eval_batch_size} \
    --learning_rate {learning_rate} \
    --gradient_accumulation_steps {gradient_accumulation_steps} \
    --num_train_epochs {num_train_epochs}"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


def make_infill_api(api_name, api_config) -> InfillAPI:
    if "_meta_info" in api_config:
        metainfo = api_config.pop("_meta_info")
        pgrsu._ilog(f"Using {metainfo['model_name']} as infill API")
        pgrsu._plog(">>>> API info", metainfo)
    cls = eval(f"{api_name}InfillAPI")
    return cls.make(**api_config)
