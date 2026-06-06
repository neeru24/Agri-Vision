from models import AnalysisHistory, BatchJob, AnalysisResult, DiseaseOccurrence, db

class AnalysisRepository:
    # AnalysisHistory
    @staticmethod
    def get_history_by_id(history_id: str) -> AnalysisHistory:
        return AnalysisHistory.query.get(history_id)

    @staticmethod
    def get_user_history(user_id: str) -> list[AnalysisHistory]:
        return AnalysisHistory.query.filter_by(user_id=user_id).order_by(AnalysisHistory.created_at.desc()).all()

    @staticmethod
    def create_history(history: AnalysisHistory) -> AnalysisHistory:
        db.session.add(history)
        db.session.commit()
        return history

    @staticmethod
    def delete_history(history: AnalysisHistory) -> None:
        db.session.delete(history)
        db.session.commit()

    # BatchJob
    @staticmethod
    def get_batch_job(job_id: str) -> BatchJob:
        return BatchJob.query.get(job_id)

    @staticmethod
    def create_batch_job(job: BatchJob) -> BatchJob:
        db.session.add(job)
        db.session.commit()
        return job

    @staticmethod
    def save_batch_job(job: BatchJob) -> BatchJob:
        db.session.commit()
        return job

    # AnalysisResult
    @staticmethod
    def get_analysis_result(result_id: str) -> AnalysisResult:
        return AnalysisResult.query.get(result_id)

    @staticmethod
    def get_batch_results(job_id: str) -> list[AnalysisResult]:
        return AnalysisResult.query.filter_by(batch_job_id=job_id).order_by(AnalysisResult.image_index).all()

    @staticmethod
    def create_analysis_result(result: AnalysisResult) -> AnalysisResult:
        db.session.add(result)
        db.session.commit()
        return result

    @staticmethod
    def save_analysis_result(result: AnalysisResult) -> AnalysisResult:
        db.session.commit()
        return result

    # DiseaseOccurrence
    @staticmethod
    def create_disease_occurrence(occurrence: DiseaseOccurrence) -> DiseaseOccurrence:
        db.session.add(occurrence)
        db.session.commit()
        return occurrence

    @staticmethod
    def expire_all() -> None:
        db.session.expire_all()

    @staticmethod
    def rollback() -> None:
        db.session.rollback()
