# sample-notebooks-causal-inference-ts-forecasting
Sample set of notebooks that explore causal inference with DoWhy &amp; Time-series forecasting with Chronos/AutoGluonTS

## Quick start
I'm into `Mamba` now, so these are the steps you can do to setup your environment.

```
$ curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
$ bash Mambaforge-$(uname)-$(uname -m).sh
```

And then run
```
$ mamba env create -f environment.yml
$ conda activate machinelearnear-ts-causal-inference
```

This will install everything you need, including `DoWhy`, `AutoGluonTS`, and then `Chronos`.

## Resources
- [Forecasting with Chronos in `AutoGluonTS`](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-chronos.html)
- [Fine-tuning scripts with Chronos](https://github.com/amazon-science/chronos-forecasting/tree/main/scripts/training)
- [`DoWhy` python library for causal inference](https://github.com/py-why/dowhy)

## License

This project is licensed under the Apache-2.0 License.