import os
import io
import time
import pickle
import _rs_utils as pgrsu
from typing import Tuple
from abc import abstractmethod


class ModelValidAPI:
    @abstractmethod
    def start(self):
        raise NotImplementedError("To be implemented")

    @abstractmethod
    def stop(self):
        raise NotImplementedError("To be implemented")

    @abstractmethod
    def valid(self, model_dir: str, out_dir: str) -> Tuple[bool, float]:
        raise NotImplementedError("To be implemented")


class AutoTrainerValidAPI(ModelValidAPI):
    __cmd_fmt = """
conda run -n {conda_env_name} python -u -W ignore {run_file} run_service \
    --name autotrainer --port {port} \
    --autotrainer-lib-path {autotrainer_lib_path}"""

    def __init__(self, conda_env_name, autotrainer_lib_path):
        self.__args = {
            "run_file": os.path.abspath(__file__),
            "port": pgrsu._available_port(),  # 37754,
            "conda_env_name": conda_env_name,
            "autotrainer_lib_path": os.path.abspath(autotrainer_lib_path),
        }

    def start(self):
        # Start the http service in other process
        import sys
        import http
        import time
        import requests
        import subprocess

        self.__proc = subprocess.Popen(
            ["bash", "-c", self.__cmd_fmt.format(**self.__args)],
            # stdout=subprocess.PIPE,
            # stderr=subprocess.PIPE,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

        pgrsu._ilog("Waiting for AutoTrainerValidAPI to be ready")
        heath_api = f"http://localhost:{self.__args['port']}/health"
        while True:
            time.sleep(5)
            try:
                if requests.post(heath_api).status_code == http.HTTPStatus.OK:
                    break
            except Exception:
                if self.__proc.poll() is not None:
                    pgrsu._flog(
                        "AutoTrainerValidAPI exited\n",
                        "stderr:\n",
                        self.__proc.stderr.read().decode("utf-8"),
                    )
                pgrsu._wlog("AutoTrainerValidAPI is not ready")
        pgrsu._ilog("AutoTrainerValidAPI is ready")

    def stop(self):
        import requests

        pgrsu._ilog("Waiting for AutoTrainerValidAPI to exit")
        exit_api = f"http://localhost:{self.__args['port']}/exit"
        requests.post(exit_api)
        self.__proc.wait()
        pgrsu._ilog("AutoTrainerValidAPI exited")

    def valid(self, model_dir: str, out_dir: str) -> Tuple[bool, float]:
        import http
        import json
        import requests

        model_dir = os.path.abspath(model_dir)
        out_dir = os.path.abspath(out_dir)
        input_j = {
            "model_dir": model_dir,
            "out_dir": out_dir,
        }
        url = f"http://localhost:{self.__args['port']}/valid"
        response = requests.post(url, data=json.dumps([input_j]))
        if response.status_code != http.HTTPStatus.OK:
            pgrsu._flog(
                "Failed to request the http API",
                response.text,
                exp=RuntimeError("Failed to request the http API"),
            )
        results = response.json()
        assert len(results) == 1
        return results[0]


class _AutoTrainerValidAPIImpl:
    def __init__(self, autotrainer_lib_path: str):
        autotrainer_lib_path = os.path.abspath(autotrainer_lib_path)
        self.autotrainer_lib = pgrsu._try_import(autotrainer_lib_path)
        self.model_train = getattr(self.autotrainer_lib, "model_train")
        assert callable(self.model_train)

    def _sfmodel_hyp_to_autotrainer_config(self, compile_hyp, fit_hyp):
        del self  # Unused

        training_config = {
            "optimizer": compile_hyp["optimizer"]["class_name"],
            "opt_kwargs": compile_hyp["optimizer"]["config"],
            "batchsize": fit_hyp["batch_size"],
            "epoch": fit_hyp["epochs"],
            "loss": compile_hyp["loss"],
            "dataset": {
                "x": fit_hyp["x"],
                "y": fit_hyp["y"],
                "x_val": (fit_hyp["validation_data"] or [fit_hyp["x"], None])[0],
                "y_val": (fit_hyp["validation_data"] or [None, fit_hyp["y"]])[1],
            },
            "callbacks": [],
        }
        return training_config

    def valid(
        self, model_dir: str, out_dir: str
    ) -> Tuple[bool, float]:  # passed, time cost
        def _set_seed(seed):
            import random, tensorflow as tf, numpy as np

            random.seed(seed)
            os.environ["PYTHONHASHSEED"] = str(seed)
            np.random.seed(seed)
            tf.random.set_seed(seed)

        import keras

        model_dir = os.path.abspath(model_dir)
        model_path = os.path.join(model_dir, "__MODEL__.h5")
        compile_hyp_path = os.path.join(model_dir, "__COMPILE_HYP__.pkl")
        fit_hyp_path = os.path.join(model_dir, "__FIT_HYP__.pkl")
        assert os.path.isfile(model_path)
        assert os.path.isfile(compile_hyp_path)
        assert os.path.isfile(fit_hyp_path)
        os.makedirs(out_dir, exist_ok=True)

        params = {  # Required by autotrainer
            "beta_1": 1e-3,
            "beta_2": 1e-4,
            "beta_3": 70,
            "gamma": 0.7,
            "zeta": 0.03,
            "eta": 0.2,
            "delta": 0.01,
            "alpha_1": 0,
            "alpha_2": 0,
            "alpha_3": 0,
            "Theta": 0.6,
        }

        root_path = out_dir  # Required by autotrainer
        save_dir = os.path.join(out_dir, "result")
        log_dir = os.path.join(out_dir, "log")
        new_issue_dir = os.path.join(out_dir, "new_issue")
        model_valid_api_log = os.path.join(out_dir, "model_valid_api.log")

        try:
            log_buffer = io.StringIO()
            with pgrsu.RedirectStdOutErrToFile(filep=log_buffer):
                try:
                    _set_seed(seed=1234)

                    start_time = None
                    model = keras.models.load_model(model_path)
                    with open(compile_hyp_path, "rb") as f:
                        compile_hyp = pickle.load(f)
                    with open(fit_hyp_path, "rb") as f:
                        fit_hyp = pickle.load(f)

                    training_config = self._sfmodel_hyp_to_autotrainer_config(
                        compile_hyp, fit_hyp
                    )
                    opt_cls = getattr(keras.optimizers, training_config["optimizer"])
                    opt = opt_cls(**training_config["opt_kwargs"])
                    batch_size = training_config["batchsize"]
                    epochs = training_config["epoch"]
                    loss = training_config["loss"]
                    dataset = training_config["dataset"]
                    callbacks = training_config["callbacks"]

                    start_time = time.time()
                    result, _, _ = self.model_train(
                        model=model,
                        train_config_set=training_config,
                        optimizer=opt,
                        loss=loss,
                        dataset=dataset,
                        iters=epochs,
                        batch_size=batch_size,
                        callbacks=callbacks,
                        verb=2,  # no progress bar
                        checktype="epoch_3",
                        autorepair=False,
                        save_dir=save_dir,
                        determine_threshold=1,
                        params=params,
                        log_dir=log_dir,
                        new_issue_dir=new_issue_dir,
                        root_path=root_path,
                    )
                    return bool(result is None), time.time() - start_time
                except Exception as ex:
                    import traceback

                    # Print full traceback with exception info
                    traceback.print_exc(file=sys.stdout)
                    if start_time is not None:
                        return False, time.time() - start_time
                    else:
                        return False, 0
        finally:
            with open(model_valid_api_log, "w") as f:
                f.write(log_buffer.getvalue())


if __name__ == "__main__":
    # Usage: python model_valid_api.py run_service --name <name> --port <port>

    import sys
    import json
    import http
    import http.server
    import argparse

    subcmd = sys.argv.pop(1)
    if subcmd == "run_service":
        parser = argparse.ArgumentParser()
        parser.add_argument("--name", type=str, required=True)
        parser.add_argument("--port", type=int, default=37754)
        args, remain_args = parser.parse_known_args()

        name = args.name
        port = args.port

        if args.name == "autotrainer":
            at_parser = argparse.ArgumentParser()
            at_parser.add_argument("--autotrainer-lib-path", type=str, required=True)
            at_args = at_parser.parse_args(remain_args)

            at_valid_api = _AutoTrainerValidAPIImpl(
                autotrainer_lib_path=at_args.autotrainer_lib_path,
            )

            class _AutoTrainerServiceHandler(http.server.BaseHTTPRequestHandler):
                def do_POST(self):
                    if self.path == "/health":
                        self.send_response(http.HTTPStatus.OK)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({"status": "ok"}).encode("utf-8"))
                    elif self.path == "/valid":
                        content_length = int(self.headers["Content-Length"])
                        body = self.rfile.read(content_length)
                        inputs = json.loads(body)
                        results = [at_valid_api.valid(**input) for input in inputs]
                        self.send_response(http.HTTPStatus.OK)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps(results).encode("utf-8"))
                    elif self.path == "/exit":
                        self.send_response(http.HTTPStatus.OK)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({"status": "ok"}).encode("utf-8"))
                        sys.exit(0)
                    else:
                        self.send_response(http.HTTPStatus.NOT_FOUND)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(
                            json.dumps({"status": "not found"}).encode("utf-8")
                        )

            httpd = http.server.HTTPServer(("", port), _AutoTrainerServiceHandler)
            httpd.serve_forever()
        else:
            raise ValueError(f"Unknown name: {name}")
    else:
        raise ValueError(f"Unknown subcmd: {subcmd}")
