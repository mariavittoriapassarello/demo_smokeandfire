from kfp.dsl import component, Output, Dataset, Model

@component(
    base_image="python:3.11",
    packages_to_install=[
        "roboflow>=1.1.28",
        "python-dotenv>=1.0.1",
        "pyyaml>=6.0.1",
        "ultralytics>=8.3.0",
    ],
)

def fetch_data(
    # ---- parametri Roboflow ----
    dataset: Output[Dataset], 
    version: str,
    rf_workspace: str = "sayed-gamall",
    rf_project: str = "fire-smoke-detection-yolov11",
    rf_version: int = 2,
    model_filename: str = "yolo11n.pt",
    workdir: str = "tmp/rf-workdir",
    rf_format: str = "yolov11",
    # ---- output KFP ----
):
    """
    Scarica dataset da Roboflow usando ROBOFLOW_KEY (da Secret),
    aggiorna data.yaml con path relativi,
    impacchetta dataset e modello come artifact KFP.
    """
    import os, sys, zipfile, yaml, shutil
    from pathlib import Path
    ROBOFLOW_KEY=os.getenv("ROBOFLOW_API_KEY")
    if not ROBOFLOW_KEY:
        raise RuntimeError(
            "ROBOFLOW_KEY non trovata. "
        )


    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    os.chdir(workdir)
    print(f"working directory: {workdir}")

    
    #download data
    from roboflow import Roboflow
    rf = Roboflow(api_key=ROBOFLOW_KEY)
    project = rf.workspace(rf_workspace).project(rf_project)
    version = project.version(rf_version)

   
    
    rf_ds = version.download(rf_format) 
    # Ricava il path locale
    if hasattr(rf_ds, "location"):
        dataset_dir = Path(rf_ds.location)
        print(f"[INFO] Dataset path: {dataset_dir}")
    elif isinstance(rf_ds, str):
        dataset_dir = Path(rf_ds)
    else:
        raise TypeError(f"not possible to find local path from: {type(rf_ds)}")


    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"file data.yaml not found in {data_yaml}")

    with open(data_yaml, "r") as f:
        data = yaml.safe_load(f)

    data["train"] = "train/images"
    data["val"] = "valid/images"
    data["test"] = "test/images"
    
    with open(data_yaml, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False)

    print("data.yaml updated with paths")
    
    dataset.path = dataset.path + ".zip"
    with zipfile.ZipFile(dataset.path, "w", zipfile.ZIP_DEFLATED) as zf:
        for entry in dataset_dir.rglob("*"):
            zf.write(entry, entry.relative_to(dataset_dir))
    print(f"Dataset in {dataset.path}")


@component(base_image='python:3.11',
           packages_to_install=["huggingface_hub"])
def fetch_model(
    model_name: str,
    version: str,
    hyperparameters: dict,
    original_model: Output[Model],
):
    try:
        import os
        import zipfile
        from pathlib import Path
        import huggingface_hub as hf
    except Exception as e:
        raise e

    HF_TOKEN: str = os.getenv("HF_TOKEN")

    # Download model checkpoint from HuggingFace repositories
    yolo_path: str = "/".join(("/tmp/", model_name))
    os.makedirs(yolo_path, exist_ok=True)

    print(f"Downloading model checkpoint: {model_name}")
    model_path = hf.snapshot_download(repo_id=model_name,
                                    allow_patterns=hyperparameters.get("checkpoint"),
                                    revision="main",
                                    token=HF_TOKEN,
                                    local_dir=yolo_path)

    # save output dataset to S3
    original_model._set_path(original_model.path + ".zip")
    srcdir = Path(yolo_path)

    with zipfile.ZipFile(original_model.path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for entry in srcdir.rglob("*"):
            zip_file.write(entry, entry.relative_to(srcdir))




    