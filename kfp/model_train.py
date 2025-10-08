from kfp.dsl import component, Input, Output, Dataset, Model, Artifact

@component(
    base_image="ultralytics/ultralytics:latest",
    packages_to_install=["pyyaml>=6.0.1"],
)
def train_model(
    dataset_zip: Input[Dataset],       
    base_model: Input[Model],         
    fine_tuned_model: Output[Model],   
    epochs: int = 1,
    imgsz: int = 640,
    batch: int = 2,
    fraction: float = 0.1
):
    import os, zipfile, shutil, time, hashlib, json
    from pathlib import Path
    from ultralytics import YOLO


    # workspace
    work = Path("/tmp/work"); work.mkdir(parents=True, exist_ok=True)
    ds_root = work / "dataset"; ds_root.mkdir(parents=True, exist_ok=True)
    model_root = work / "base_model"; model_root.mkdir(parents=True, exist_ok=True)
    runs_root = work / "runs"; runs_root.mkdir(parents=True, exist_ok=True)

    # Dataset: unzip
    ds_src = Path(dataset_zip.path)
    print(f"[INFO] Input dataset path: {ds_src}")
    if ds_src.is_file() and ds_src.suffix == ".zip":
        with zipfile.ZipFile(ds_src, "r") as z:
            z.extractall(ds_root)
        print(f"[INFO] Dataset estratto in: {ds_root}")
    elif ds_src.is_dir():
        shutil.copytree(ds_src, ds_root, dirs_exist_ok=True)
        print(f"[INFO] Dataset copiato in: {ds_root}")
    else:
        raise ValueError(f"Formato dataset non supportato: {ds_src}")

    data_yaml = next(ds_root.rglob("data.yaml"), None)
    if not data_yaml:
        raise FileNotFoundError("data.yaml non trovato nel dataset estratto")
    print(f"[INFO] Using data.yaml: {data_yaml}")

    # 
    msrc = Path(base_model.path)
    print(f"[INFO] Base model artifact: {msrc}")

    pt_candidates = []
    if msrc.is_file() and msrc.suffix == ".zip":
        with zipfile.ZipFile(msrc, "r") as z:
            z.extractall(model_root)
        pt_candidates = sorted(model_root.rglob("*.pt"))
    elif msrc.is_file() and msrc.suffix == ".pt":
        pt_candidates = [msrc]
    elif msrc.is_dir():
        pt_candidates = sorted(msrc.rglob("*.pt"))
    else:
        raise ValueError(f"Atteso ZIP/.pt/dir per il modello, trovato: {msrc}")

    if not pt_candidates:
        raise FileNotFoundError("Nessun file .pt trovato nel modello base")
    weights_path = str(pt_candidates[0])
    print(f"[INFO] Using base weights: {weights_path}")

    # Training 
    model = YOLO(weights_path)
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        fraction=fraction,
        project=str(runs_root),  
        name="train",             
        exist_ok=True,
    )

    #Individua la cartella risultati reale
    save_dir = None
    if getattr(results, "save_dir", None):
        save_dir = Path(results.save_dir)
    elif getattr(getattr(model, "trainer", None), "save_dir", None):
        save_dir = Path(model.trainer.save_dir)
    else:
        save_dir = runs_root / "train"  

    print(f"Ultralytics save_dir: {save_dir}")

    # Trova best/last sotto save_dir/weights
    weights_dir = save_dir / "weights"
    best = None
    if weights_dir.exists():
        c_best = sorted(weights_dir.glob("best*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        c_last = sorted(weights_dir.glob("last*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        best = (c_best + c_last)[0] if (c_best or c_last) else None

    if not (best and best.exists()):
        # log di supporto per debugging
        print(f"Nessun best.pt/last.pt trovato in {weights_dir}")
        print("Contenuto save_dir:")
        for p in save_dir.rglob("*"):
            if p.is_file():
                print(" -", p)
        raise FileNotFoundError(f"Nessun best.pt/last.pt trovato in {weights_dir}")
   
        
    #zip model
    model_zip_path = Path(str(fine_tuned_model.path) + ".zip")
    model_zip_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(model_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # includi almeno il best.pt (puoi aggiungere altro se vuoi)
        zf.write(best, arcname="best.pt")

    # reindirizza l'artifact al file zip
    fine_tuned_model._set_path(str(model_zip_path))


    if not model_zip_path.exists() or model_zip_path.stat().st_size == 0:
        raise RuntimeError(f"Model zip non scritto correttamente: {model_zip_path}")
    print(f"[SUCCESS] Model zipped: {model_zip_path} size={model_zip_path.stat().st_size}")

    # Metadati utili
    fine_tuned_model.metadata["framework"]= "ultralytics"
    fine_tuned_model.metadata["format"]= "pt"
    fine_tuned_model.metadata["compression"]= "zip"
    fine_tuned_model.metadata["filename"]= "best.pt"


    # Output: best.pt alla root dell'artifact
    #out_dir = Path(fine_tuned_model.path); out_dir.mkdir(parents=True, exist_ok=True)
    #shutil.copy2(best, out_dir / "best.pt")
    #print(f"[SUCCESS] Saved model to: {out_dir / 'best.pt'}")
    







