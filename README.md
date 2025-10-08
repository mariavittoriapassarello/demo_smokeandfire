WIP

Fine-tuning of a YOLOv11 model on Red Hat OpenShift AI 2.23. The training has been done on Smoke and Fire images belonging to a public dataset downloaded from Roboflow (at https://universe.roboflow.com/sayed-gamall/fire-smoke-detection-yolov11) 


**To run this demo on Red Hat OpenShift AI, the following components are required:**
- Model Registry enabled and set up
- S3 storage (MinIO in this case) with dedicated buckets for pipelines, artifacts, and models.
- When running Elyra pipelines, ensure the S3 connection to the models bucket is attached to the workbench.
- The following secrets must be created:
  - Roboflow API key (for dataset download)
  - Hugging Face credentials (for YOLOv11 model download)





Improvements to be made 
- enhance pipeline with kfp:
  - remove cached results
  - add pipeline versioning
  - ...
- enable GPU execution of the kfp pipeline
- generate training metrics, graphs, and plots as pipeline artifacts for visualization in the OpenShift AI dashboard.
- try Kubeflow Trainer
- integrate bias and drift monitoring with Trusty AI
