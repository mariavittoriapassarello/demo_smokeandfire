WIP

Fine-tuning of a YOLOv11 model on Red Hat OpenShift AI 2.23. The training has been done on Smoke and Fire images belonging to a public dataset downloaded from Roboflow (at https://universe.roboflow.com/sayed-gamall/fire-smoke-detection-yolov11) 

**For this demo, in Red Hat OpenShift AI it is needed to set up:**
- Model Registry 
- s3 (Minio in this case) with buckets for pipelines, artifacts and models.
- to execute Elyra pipelines, s3 connection to bucket models needs to be attached to the workbench
- the following secrets have to be created: Roboflow API key for downloading the dataset from Roboflow, HuggingFace credentials to download the YOLOv11 model 


Improvements to be made 
- pipeline with kfp needs to be updated and improved (remove cached results, add versioning of pipeline,...)
- running kfp pipeline on GPU
- generate fine-tuning graphs and plots as pipeline artifacts to be displayed on OpenShift AI dashboard
- try kft operator
- try monitoring for bias and drift 
