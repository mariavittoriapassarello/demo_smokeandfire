from kfp.dsl import (
    component,
    Input,
    Dataset,
    Metrics,
    Model,
    Artifact,
)


@component(base_image='python:3.11',
           packages_to_install=['pip==24.2',
                                'setuptools==74.1.3',
                                'boto3==1.36.12',
                                'model-registry'])


def push_to_s3_and_model_registry(
    onnx_model: Input[Model],   
    version: str,
    registered_model_name: str = "smokeandfire",
    version_to_mr: str = "0.0.8",
    username: str = "Maria Vittoria Passarello",
    cluster_domain: str = "apps.cluster-2lxjg.2lxjg.sandbox2810.opentlc.com",
    description: str = "YOLO model fine tuned on images of smoke and fire",
    accuracy: float = 0.55,
    fraction: float = 0.1,
    epoch: int = 1,
    license_name: str = "apache-2.0",
    is_secure: bool = False, 
):


    import os, time, json
    from pathlib import Path
    from model_registry import ModelRegistry
    from model_registry.utils import S3Params
    from model_registry.exceptions import StoreError
    from os import environ
    
    
    # look for model 
    
    root = Path(onnx_model.path)
    onnx_path = root / "model.onnx"
    if not onnx_path.exists():
        try:
            onnx_path = next(root.rglob("*.onnx"))
        except StopIteration:
            raise FileNotFoundError(f"Nessun file .onnx trovato in {root}")

    #get s3 credentials
    AWS_S3_ENDPOINT      = os.environ.get("AWS_S3_ENDPOINT", "")
    AWS_ACCESS_KEY_ID    = os.environ.get("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY= os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    AWS_S3_BUCKET        = os.environ.get("AWS_S3_BUCKET", "")
    
    if not AWS_S3_BUCKET:
        raise RuntimeError("AWS_S3_BUCKET not found")
        
    environ["KF_PIPELINES_SA_TOKEN_PATH"] = "/var/run/secrets/kubernetes.io/serviceaccount/token"


    s3_prefix = f"{registered_model_name}/{version}"
    model_registry_url = f"https://model-registry1-rest.{cluster_domain}"
    registry = ModelRegistry(server_address=model_registry_url, port=443, author=username, is_secure=is_secure)

    minio_endpoint = "https://minio-api-minio.apps.cluster-2lxjg.2lxjg.sandbox2810.opentlc.com"

    
    s3_upload_params = S3Params(
        bucket_name=AWS_S3_BUCKET,
        s3_prefix=s3_prefix,
    )

    
    url_dashboard_base = f"https://rhods-dashboard-redhat-ods-applications.{cluster_domain}"
    created, link, version_id = False, "", None

    try:
        registered_model = registry.upload_artifact_and_register_model(
            name=registered_model_name,
            model_files_path=str(onnx_path),
            model_format_name="onnx",
            model_format_version="1",
            author=username,
            version=version_to_mr,
            description=description,
            metadata={
                "accuracy": accuracy,
                "fraction": fraction,
                "epoch": epoch,
                "license": license_name,
            },
            upload_params=s3_upload_params,
        )
        created = True
        mv = registry.get_model_version(registered_model_name, version_to_mr)
        version_id = getattr(mv, "id", None)
        if version_id is not None:
            link = f"{url_dashboard_base}/modelRegistry/{username}-registry/registeredModels/1/versions/{version_id}/details"
    except StoreError:
        mv = registry.get_model_version(registered_model_name, version_to_mr)
        version_id = getattr(mv, "id", None)
        if version_id is not None:
            link = f"{url_dashboard_base}/modelRegistry/{username}-registry/registeredModels/1/versions/{version_id}/details"
   
