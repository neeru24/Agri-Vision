import os
import logging
from datetime import datetime
import base64

logger = logging.getLogger(__name__)

MAX_BATCH_SIZE = int(os.getenv("AGRI_MAX_BATCH_SIZE", "200"))
CELERY_BROKER = os.getenv("REDIS_URL", "redis://localhost:6379/0")

try:
    from celery import Celery, group, chord

    celery = Celery("agri_vision", broker=CELERY_BROKER, backend=CELERY_BROKER)
    celery.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
        task_time_limit=300,
        task_soft_time_limit=240,
    )

    CELERY_AVAILABLE = True
except Exception:
    celery = None
    CELERY_AVAILABLE = False


def _ensure_app_context():
    from app import app

    return app


if CELERY_AVAILABLE:
    @celery.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
    def analyze_image_task(self, job_id: str, result_id: str, image_name: str, image_index: int, image_b64: str):
        """Analyse one image and update DB AnalysisResult row."""
        import cv2
        import numpy as np

        app = _ensure_app_context()
        from app import analyze_image
        from models import AnalysisResult, db, BatchJob

        image_name = "unknown"
        image_index = 0

        # Retrieve image info from pre-created AnalysisResult row
        try:
            with app.app_context():
                res_row = db.session.get(AnalysisResult, result_id) if hasattr(db.session, 'get') else AnalysisResult.query.get(result_id)
                if res_row:
                    image_name = res_row.image_name
                    image_index = res_row.image_index
        except Exception as query_err:
            logger.error(f"Error querying AnalysisResult {result_id}: {query_err}")

        try:
            file_bytes = base64.b64decode(image_b64)
            arr = np.frombuffer(file_bytes, np.uint8)
            image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Invalid image data")

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = analyze_image(image_rgb)
            
            # Save results to database
            try:
                with app.app_context():
                    result = db.session.get(AnalysisResult, result_id) if hasattr(db.session, 'get') else AnalysisResult.query.get(result_id)
                    if result:
                        result.status = "complete"
                        result.disease_class = results.get("disease", {}).get("predicted_class")
                        result.disease_confidence = results.get("disease", {}).get("confidence")
                        result.health_score = results.get("disease", {}).get("health_score")
                        result.growth_class = results.get("growth", {}).get("main_class")
                        result.growth_confidence = results.get("growth", {}).get("confidence")
                        result.results_json = results
                        
                        job = db.session.get(BatchJob, job_id) if hasattr(db.session, 'get') else BatchJob.query.get(job_id)
                        if job:
                            completed_count = len([r for r in job.results if r.status in ("complete", "success")])
                            failed_count = len([r for r in job.results if r.status == "error"])
                            if completed_count + failed_count + 1 >= job.total_images:
                                job.status = "completed"
                                job.completed_at = datetime.utcnow()
                        db.session.commit()
            except Exception as db_err:
                logger.error(f"Error saving analysis result to database: {db_err}")
            
            # Update task status to complete
            self.update_state(
                state='SUCCESS',
                meta={
                    'job_id': job_id,
                    'image_index': image_index,
                    'status': 'complete',
                    'results': results
                }
            )
            
            return {
                'image_name': image_name,
                'image_index': image_index,
                'status': 'complete',
                'results': results,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing image {image_name}: {e}")
            self.update_state(
                state='FAILURE',
                meta={
                    'job_id': job_id,
                    'image_index': image_index,
                    'status': 'error',
                    'error': str(e)
                }
            )
            try:
                with app.app_context():
                    result = db.session.get(AnalysisResult, result_id) if hasattr(db.session, 'get') else AnalysisResult.query.get(result_id)
                    if result:
                        result.status = "error"
                        result.error_message = str(e)
                        
                        job = db.session.get(BatchJob, job_id) if hasattr(db.session, 'get') else BatchJob.query.get(job_id)
                        if job:
                            completed_count = len([r for r in job.results if r.status in ("complete", "success")])
                            failed_count = len([r for r in job.results if r.status == "error"])
                            if completed_count + failed_count + 1 >= job.total_images:
                                job.status = "completed"
                                job.completed_at = datetime.utcnow()
                        db.session.commit()
            except Exception as db_err:
                logger.error(f"Error saving failure result to database: {db_err}")
            raise


    @celery.task(bind=True)
    def finalize_batch_job(self, job_id: str, results_meta: list):
        from models import BatchJob, AnalysisResult, db

        app = _ensure_app_context()
        with app.app_context():
            job = BatchJob.query.get(job_id)
            if not job:
                logger.error("finalize_batch_job: job not found %s", job_id)
                return {"job_id": job_id, "status": "missing"}

            completed = AnalysisResult.query.filter_by(batch_job_id=job_id, status="complete").count()
            failed = AnalysisResult.query.filter_by(batch_job_id=job_id, status="error").count()
            job.completed_images = completed
            job.failed_images = failed
            job.status = "completed" if (completed + failed) >= job.total_images else "processing"
            if job.status == "completed":
                job.completed_at = datetime.utcnow()
            db.session.commit()

            return {"job_id": job_id, "status": job.status, "completed": completed, "failed": failed}


    @celery.task(bind=True)
    def process_batch_job(self, job_id: str, images_data: list):
        from models import BatchJob, AnalysisResult, db

        app = _ensure_app_context()
        with app.app_context():
            job = BatchJob.query.get(job_id)
            if not job:
                raise ValueError(f"Batch job {job_id} not found")

            if len(images_data) > MAX_BATCH_SIZE:
                raise ValueError(f"Batch size exceeds maximum of {MAX_BATCH_SIZE}")

            job.status = "processing"
            job.started_at = datetime.utcnow()
            job.total_images = len(images_data)
            db.session.commit()

            task_sigs = []
            for idx, (image_name, b64) in enumerate(images_data):
                res = AnalysisResult(batch_job_id=job.id, image_name=image_name, image_index=idx, status="pending")
                db.session.add(res)
                db.session.flush()

                sig = analyze_image_task.s(job.id, res.id, b64)
                task_sigs.append(sig)

            db.session.commit()

            job.task_ids = []
            db.session.commit()

            group_obj = group(task_sigs)
            chord(group_obj)(finalize_batch_job.s(job.id))

            return {"job_id": job.id, "dispatched": len(task_sigs)}

else:
    def analyze_image_task(*args, **kwargs):
        raise NotImplementedError("Celery is not available in this environment")


    def process_batch_job(job_id: str, images_data: list):
        import concurrent.futures
        import threading
        import cv2
        import numpy as np
        from app import analyze_image
        from models import BatchJob, AnalysisResult, db

        if len(images_data) > MAX_BATCH_SIZE:
            raise ValueError(f"Batch size exceeds maximum of {MAX_BATCH_SIZE}")

        def _worker():
            from app import app
            with app.app_context():
                job = BatchJob.query.get(job_id)
                if not job:
                    logger.error("Fallback worker: job not found %s", job_id)
                    return
                job.status = "processing"
                job.started_at = datetime.utcnow()
                job.total_images = len(images_data)
                db.session.commit()

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
                futures = []
                for idx, (image_name, b64) in enumerate(images_data):
                    futures.append(ex.submit(_process_single, job_id, idx, image_name, b64))

                for fut in concurrent.futures.as_completed(futures):
                    try:
                        fut.result()
                    except Exception as e:
                        logger.exception("Fallback image processing error: %s", e)

            from app import app
            with app.app_context():
                job = BatchJob.query.get(job_id)
                completed = AnalysisResult.query.filter_by(batch_job_id=job_id, status="complete").count()
                failed = AnalysisResult.query.filter_by(batch_job_id=job_id, status="error").count()
                job.completed_images = completed
                job.failed_images = failed
                job.status = "completed"
                job.completed_at = datetime.utcnow()
                db.session.commit()

        def _process_single(job_id, idx, image_name, b64):
            from app import analyze_image
            from models import AnalysisResult, db
            import base64
            import numpy as np
            import cv2

            app = _ensure_app_context()
            with app.app_context():
                res = AnalysisResult(batch_job_id=job_id, image_name=image_name, image_index=idx, status="pending")
                db.session.add(res)
                db.session.commit()

            try:
                raw = base64.b64decode(b64)
                arr = np.frombuffer(raw, np.uint8)
                image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("Invalid image data")
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = analyze_image(image_rgb)

                with app.app_context():
                    r = AnalysisResult.query.get(res.id)
                    r.status = "complete"
                    r.results_json = results
                    disease = results.get("disease", {})
                    r.disease_class = disease.get("predicted_class")
                    r.disease_confidence = disease.get("confidence")
                    r.health_score = disease.get("health_score")
                    growth = results.get("growth", {})
                    r.growth_class = growth.get("main_class")
                    r.growth_confidence = growth.get("confidence")
                    db.session.commit()
            except Exception as e:
                logger.exception("Fallback processing failed for %s: %s", image_name, e)
                with app.app_context():
                    r = AnalysisResult.query.get(res.id)
                    r.status = "error"
                    r.error_message = str(e)
                    db.session.commit()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        return {"job_id": job_id, "status": "started_fallback"}
