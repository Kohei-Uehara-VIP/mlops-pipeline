from mlflow import MlflowClient
client = MlflowClient()


client.set_registered_model_alias(
    name="WineQualityModel",
    alias="staging",
    version=4
)
