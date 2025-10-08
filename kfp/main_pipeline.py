import kfp

#import objects 
import kfp.dsl as dsl
from kfp.dsl import pipeline
from kfp import kubernetes

#import components 
from fetch_data_and_mod import fetch_data, fetch_model
from model_train import train_model
from convert_model_to_onnx import convert_to_onnx
from save_model_to_s3_and_model_registry import push_to_s3_and_model_registry

roboflow_api_key = 'roboflow'
huggingface_secret= 'huggingface'
models = 's3-models'


#pipeline definition

#create pipeline
@pipeline(name="yolo-custom-training-pipeline-test", description="Dense Neural Network Image Detector based on YOLO",
)
def training_pipeline(hyperparameters: dict,
                      version: str,


):
    #fetch data
    fetch_dataset_task = fetch_data(rf_workspace="sayed-gamall", version = version, rf_project="fire-smoke-detection-yolov11")
    kubernetes.use_secret_as_env(
        fetch_dataset_task,
        secret_name=roboflow_api_key,
        secret_key_to_env={
            "ROBOFLOW_API_KEY": "ROBOFLOW_API_KEY",
        },
    )
    fetch_model_task = fetch_model(model_name="Ultralytics/YOLO11", version=version, hyperparameters=hyperparameters)
    kubernetes.use_secret_as_env(
        fetch_model_task,
        secret_name=huggingface_secret,
        secret_key_to_env={
            "HF_NAME": "HF_NAME",
            "HF_TOKEN": "HF_TOKEN"
        },
    )

    
    #train model

    train_model_task = train_model(base_model=fetch_model_task.outputs["original_model"], 
                                   dataset_zip=fetch_dataset_task.outputs["dataset"])
    #convert fine tuned model to onnx
    convert_model_task = convert_to_onnx(fine_tuned_model_zip=train_model_task.outputs["fine_tuned_model"])
    convert_model_task.after(train_model_task)

    #save model to s3 and model registry
    save_task = push_to_s3_and_model_registry(onnx_model = convert_model_task.outputs["onnx_model"], version = version)
    save_task.after(convert_model_task)

    kubernetes.use_secret_as_env(save_task, secret_name=models, secret_key_to_env={
        "AWS_ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY",
        "AWS_S3_ENDPOINT": "AWS_S3_ENDPOINT",
        "AWS_S3_BUCKET": "AWS_S3_BUCKET",
    },
)



# start pipeline
if __name__ == "__main__":
    metadata = {
        "hyperparameters": {
            "epochs": 1,
            "batch": 2,
            "img_size": 640,
            "learning_rate": 1e-4,
            "batch_size": 128,
            "job": "detect",
            "run_name": "train",
            "checkpoint": "yolo11x.pt",
            "optimizer": "AdamW",
            "augment": True,
            "fraction": 0.1
        },
        "version": "8"
    }
    namespace_file_path = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
    with open(namespace_file_path, "r") as namespace_file:
        namespace = namespace_file.read()

    kubeflow_endpoint = f"https://ds-pipeline-dspa.{namespace}.svc:8443"


    sa_token_file_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
    with open(sa_token_file_path, "r") as token_file:
        bearer_token = token_file.read()
    ssl_ca_cert = "/var/run/secrets/kubernetes.io/serviceaccount/service-ca.crt"

    from kfp import compiler

    compiler.Compiler().compile(training_pipeline, "pipeline.yaml")

    # Run pipeline on cluster
    print(f"Connecting to Data Science Pipelines: {kubeflow_endpoint}")
    client = kfp.Client(
        host=kubeflow_endpoint, existing_token=bearer_token, ssl_ca_cert=ssl_ca_cert
    )

    client.create_run_from_pipeline_func(
        training_pipeline, arguments=metadata, 
        experiment_name="yolo-custom-training-pipeline",
        enable_caching=True,
    )