"""
Celery tasks for batch image processing
"""
import os
import uuid
import logging
from datetime import datetime

# Lazy import Celery to avoid errors if not installed
try:
    from celery import Celery
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    celery = Celery('agri_vision', broker=redis_url, backend=redis_url)
    celery.conf.update(
        task_serializer='json',
        accept_content=['json'],
        result_serializer='json',
        timezone='UTC',
        enable_utc=True,
        task_track_started=True,
        task_time_limit=300,  # 5 minutes per task
    )
    CELERY_AVAILABLE = True
except ImportError:
    celery = None
    CELERY_AVAILABLE = False

logger = logging.getLogger(__name__)


# Define task decorators that work with or without Celery
if CELERY_AVAILABLE:
    @celery.task(bind=True)
    def analyze_image_task(self, image_data, image_name, job_id, image_index):
        """
        Celery task to analyze a single image
        """
        import cv2
        import numpy as np
        from app import analyze_image, model_manager
        
        try:
            # Update task status
            self.update_state(
                state='PROGRESS',
                meta={'job_id': job_id, 'image_index': image_index, 'status': 'processing'}
            )
            
            # Decode image
            file_bytes = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Invalid image file: {image_name}")
            
            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Analyze image
            results = analyze_image(image_rgb)
            
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
            raise

    @celery.task
    def process_batch_job(job_id, images_data):
        """
        Orchestrates batch processing of multiple images
        """
        from models import BatchJob, db
        
        try:
            # Get batch job from database
            job = BatchJob.query.get(job_id)
            if not job:
                raise ValueError(f"Batch job {job_id} not found")
            
            # Update job status
            job.status = 'processing'
            job.started_at = datetime.utcnow()
            db.session.commit()
            
            # Create tasks for each image
            task_ids = []
            for idx, (image_name, image_data) in enumerate(images_data):
                task = analyze_image_task.delay(image_data, image_name, job_id, idx)
                task_ids.append(task.id)
            
            # Store task IDs in job
            job.task_ids = task_ids
            db.session.commit()
            
            return {'job_id': job_id, 'task_count': len(task_ids)}
            
        except Exception as e:
            logger.error(f"Error starting batch job {job_id}: {e}")
            if job:
                job.status = 'failed'
                job.error_message = str(e)
                db.session.commit()
            raise
else:
    # Stub functions when Celery is not available
    def analyze_image_task(*args, **kwargs):
        raise NotImplementedError("Celery is not installed. Install with: pip install celery")
    
    def process_batch_job(*args, **kwargs):
        raise NotImplementedError("Celery is not installed. Install with: pip install celery")
