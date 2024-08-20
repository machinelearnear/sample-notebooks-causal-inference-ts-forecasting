# sample-notebooks-causal-inference-ts-forecasting
Sample set of notebooks that explore causal inference with DoWhy &amp; Time-series forecasting with Chronos/AutoGluonTS

## Quick start
I'm into `Mamba` now, so these are the steps you can do to setup your environment.

```bash
$ curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
$ bash Mambaforge-$(uname)-$(uname -m).sh
```

And then run
```bash
$ mamba env create -f environment.yml
$ conda activate machinelearnear-ts-causal-inference
```

This will install everything you need, including `DoWhy`, `AutoGluonTS`, and then `Chronos`.

Additionally, you could just use your own Python environment and then run

```bash
$ pip install -r requirements.txt
```

## Deploy `chronos-t5` models

:rocket: Chronos pre-trained models for time series forecasting are now available on SageMaker JumpStart! :rocket:
[Chronos](https://github.com/amazon-science/chronos-forecasting?tab=readme-ov-file) is a family of time series forecasting models developed by Amazon. Chronos models are pre-trained on a large collection of open-source time series data and can generate accurate probabilistic & point forecasts in zero-shot manner.
JumpStart makes it easy to deploy production-ready endpoints serving `chronos-t5-small`, `chronos-t5-base` or `chronos-t5-large` models through either `SageMaker Studio` or `SageMaker SDK`.

Example deploying a chronos-t5-base endpoint via SageMaker Python SDK:

```python
from sagemaker.jumpstart.model import JumpStartModel

model = JumpStartModel(
    model_id="autogluon-forecasting-chronos-t5-base",
    role="AmazonSageMaker-ExecutionRole-XXXXXXXXXXXXXXX",  # replace with your SageMaker execution role
)

predictor = model.deploy(
    instance_type="g5.xlarge",  # single-GPU g5, p3, g4dn instances supported
)
```

Example forecasting two univariate time series:
```python
payload = {
    "inputs": [
        {"target": [1.0, 2.0, 3.0, 2.0, 0.5, 2.0, 3.0, 2.0, 1.0], "item_id": "product_A"},
        {"target": [5.4, 3.0, 3.0, 2.0, 1.5, 2.0, -1.0], "item_id": "product_B"},
    ],
    "parameters": {
        "prediction_length": 5,
    }
}

response = predictor.predict(payload)
```

Check out the SageMaker Studio webpage for Chronos models for more information on model usage.
The models are already live and ready to use both via SageMaker Studio and the SageMaker SDK!

## Benchmarking
This is a nice article by Hachi titled ["Trying out time-series-based models on Google Colab ④: Amazon Chronos-T5"](https://note-com.translate.goog/hatti8/n/n9e9221c8d1ca?_x_tr_sl=ja&_x_tr_tl=en&_x_tr_hl=en&_x_tr_pto=wapp&_x_tr_hist=true). It's benchmark across several commercially available licensed time-series-based models from HuggingFace.

**`google/timesfm-1.0-200m`**
- Download count: 4.59k
- Model size: 200m
- License: Apache-2.0

**`AutonLab/MOMENT-1-large`**
- Download count: 5.79k
- Model size: 385m
- License: MIT

**`ibm-granite/granite-timeseries-ttm-v1`**
- Download count: 10.1k
- Model size: 805k ( small!! )
- License: Apache-2.0

**`amazon/chronos-t5-large`** (this time)
- Number of downloads: 256k ( a lot! )
- Model size: 709m
- License: Apache-2.0

## Resources
- [Forecasting with Chronos in `AutoGluonTS`](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-chronos.html)
- [Fine-tuning scripts with Chronos](https://github.com/amazon-science/chronos-forecasting/tree/main/scripts/training)
- [`DoWhy` python library for causal inference](https://github.com/py-why/dowhy)
- [Cloud Training and Deployments with Amazon SageMaker](https://auto.gluon.ai/stable/tutorials/cloud_fit_deploy/cloud-aws-sagemaker-train-deploy.html)
- [`autogluon.timeseries.TimeSeriesPredictor`](https://auto.gluon.ai/stable/api/autogluon.timeseries.TimeSeriesPredictor.html)

## License

This project is licensed under the Apache-2.0 License.