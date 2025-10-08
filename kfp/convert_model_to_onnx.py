from kfp.dsl import component, Input, Output, Model

@component(
    base_image="ultralytics/ultralytics:latest",  
    packages_to_install=["onnx>=1.14.0",
        "onnxruntime>=1.17.0",         
        "onnxslim>=0.1.67",            
        "onnxsim>=0.4.36",  ]                        
)
def convert_to_onnx(
    fine_tuned_model_zip: Input[Model],  
    onnx_model: Output[Model],       
):
    from pathlib import Path
    from shutil import copy2
    from ultralytics import YOLO
    import zipfile

    
    #prendo file zip
    model_zip_path = Path(fine_tuned_model_zip.path)
    if not model_zip_path.exists():
        raise FileNotFoundError(f"File zip del modello non trovato: {model_zip_path}")
        
    #dir temporanea
    
    work = Path("/tmp/work"); work.mkdir(parents=True, exist_ok=True)
    extract_dir = work / "unzipped_model"
    extract_dir.mkdir(parents=True, exist_ok=True)

    #unzip
    with zipfile.ZipFile(model_zip_path, "r") as zf:
        zf.extractall(extract_dir)
        print(f"[INFO] Estratto {len(zf.namelist())} file da {model_zip_path} in {extract_dir}")

    # Trova il file best.pt (o last.pt se non presente)
    weights = next(extract_dir.rglob("best.pt"), None)
    if not weights:
        weights = next(extract_dir.rglob("last.pt"), None)
    if not weights or not weights.exists():
        raise FileNotFoundError(f"Nessun file .pt trovato nello zip estratto in {extract_dir}")
    print(f"[INFO] Peso trovato: {weights}")

    export_dir = work / "runs" / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    # esporta in formato ONNX
    print("Esportazione da YOLO.pt a ONNX")
    result = YOLO(str(weights)).export(
        format="onnx",
        project=str(export_dir),
        name="onnx",
        exist_ok=True,
    )

    # trova file .onnx 
    onnx_path = None
    if result and Path(result).exists():
        onnx_path = Path(result)
    else:
        candidates = list((export_dir / "onnx").rglob("*.onnx"))
        if not candidates:
            raise FileNotFoundError("File .onnx non trovato dopo l’esportazione")
        onnx_path = candidates[0]
    print(f" Modello ONNX generato: {onnx_path} ({onnx_path.stat().st_size} bytes)")

    # copia il file ONNX nell’output artifact
    out_dir = Path(onnx_model.path)
    out_dir.mkdir(parents=True, exist_ok=True)
    copy2(onnx_path, out_dir / "model.onnx")

    # metadata
    onnx_model.metadata["framework"] = "ultralytics"
    onnx_model.metadata["format"] = "onnx"
    onnx_model.metadata["source"] = str(weights)
    onnx_model.metadata["compression"] = "unzipped"

    print(f"Model saved to: {out_dir / 'model.onnx'}")


